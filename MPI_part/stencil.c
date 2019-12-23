#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

#define MASTER 0
#define OUTPUT_FILE "stencil.pgm"
#define tag 122
#define OUTPUT_BEGINING "initial.pgm"
#define NDIMS 1
#define MASTER 0

int calc_nrows_from_rank(int rank, int size, int ny);
void output_image(const char * file_name, const int nx, const int ny, float * restrict image);
void init_image(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image, int firstrow, int lastrow, float * restrict sendbuf, float * restrict recvbuf, int above, int below, MPI_Status status, int rank, int size);
double wtime(void);
int Local_col(const int cols, const int size, const int rank);
void param(const int rows,const int cols,const int size, int *displs, int *sendcounts);

int Local_col(const int cols, const int size, const int rank){
	int local_cols = cols/size;
	int remain  = cols%size;
	if (remain != 0 && rank < remain) local_cols++;
	return local_cols;
}
void param(const int rows,const int cols,const int size, int *displs, int *sendcounts){

	int rank;
	int rank_cols;
	for(int rank=0; rank < size; rank++){
		rank_cols = Local_col(cols,size,rank);
		// add this to displs[] array
		sendcounts[rank] = rank_cols * rows;
	}
	for(rank=0; rank < size; rank++){
		if (rank == 0) displs[rank] = 0;
		else {
			displs[rank] = displs[rank-1] + sendcounts[rank-1];
		}
	}
}
int main(int argc, char* argv[]) {

		// Check usage
		if (argc != 4) {
				fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
				exit(EXIT_FAILURE);
		}
		MPI_Status status;
		int *displs;
		int *sendcounts;
		int nx = atoi(argv[1]);
		int ny = atoi(argv[2]);
		int niters = atoi(argv[3]);
		int rows = ny +2;
		int cols = nx;
		int above;
		int below;
		int size;
		int rank;
		int reorder = 1;
		int dims[NDIMS];
		int periods[NDIMS];
		MPI_Comm CART_COMM_WORLD;
		for (int i=0; i<NDIMS; i++) {
			dims[i] = 0;
			periods[i] = 0;
		}

		MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Dims_create(size, NDIMS, dims);
		//printf("help before caet\n" );

		MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, reorder, &CART_COMM_WORLD);
		MPI_Comm_rank(CART_COMM_WORLD, &rank );

		int local_nrows = calc_nrows_from_rank(rank, size, ny);
		int local_ncols = nx;
		float *image;
		float *tmp_image;
		float *sendbuf;
		float *recvbuf;
		int firstrow = rank * local_nrows;
		int lastrow;

		//calculate the starting and ending rows
		if (rank == size - 1) {
				lastrow = ny - 1;
		} else {
				lastrow = firstrow + local_nrows - 1;
		}

		//calculate the rank above and below
		above = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
		below = (rank + 1) % size;

		//allocate memory for images and bufferers
		image = malloc(sizeof(float) * nx * ny);
		tmp_image = malloc(sizeof(float) * nx * ny);
		float* result_image = malloc(sizeof(float) * nx * ny);

		int local_rows = ny+2;
		int local_cols = cols/size+2;
		int sectionSize = local_rows*local_cols;
		float * local_image = malloc(sizeof(float) * sectionSize);
		float * localTmp = malloc(sizeof(float) * sectionSize);
		memset (local_image, 0, sizeof(float) * sectionSize);
		memset (localTmp, 0, sizeof(float) * sectionSize);

		sendbuf = malloc(sizeof(float) * (nx+2)*2);
		recvbuf = malloc(sizeof(float) * (nx+2)*2);
		memset(recvbuf,0,sizeof(float)*(nx+2)*2);
		memset(sendbuf,0,sizeof(float)*(nx+2)*2);
		init_image(nx, ny, image, tmp_image);

		displs = malloc(sizeof(int)*size);
		sendcounts = malloc(sizeof(int)*size);
		param(rows, cols, size, displs, sendcounts);

		double toc, tic;
		if (size != 1) {
				tic = wtime();
				for (int t = 0; t < niters; t++) {
						stencil(nx, ny, image, tmp_image, firstrow, lastrow, sendbuf, recvbuf, above, below, status, rank, size);
						stencil(nx, ny, tmp_image, image, firstrow, lastrow, sendbuf, recvbuf, above, below, status, rank, size);
				}
				toc = wtime();
		} else {
				tic = wtime();
				for (int t = 0; t < niters; t++) {
					tmp_image[0] = image[0] * 0.6f + (image[ny] + image[1]) * 0.1f;

					//top row
					for(int i = 1; i < ny - 1; ++i){
							tmp_image[i] = image[i] * 0.6f + (image[i - 1] + image[i + 1] + image[ny + i]) * 0.1f;
					}

					//top right cell
					tmp_image[ny-1] = image[ny-1] * 0.6f + (image[ny-2] + image[2*  ny-1]) * 0.1f;

					//left side column
					for(int j = 1; j < ny - 1; ++j){
							tmp_image[ny * j] = image[ny * j] * 0.6f + (image[ny * (j - 1)] + image[ny * (j + 1)] + image[ny * j + 1]) * 0.1f;
					}

					//right side column
					for(int j = 1; j < ny - 1; ++j){
							tmp_image[ny * j + ny - 1] = image[ny * j + ny - 1] * 0.6f + (image[ny * (j - 1) + ny - 1] + image[ny * (j + 1) + ny - 1] + image[ny * j + ny - 2]) * 0.1f;
					}

					//inner grid
					for (int i = 1; i < ny - 1; ++i) {
							for (int j = 1; j < nx - 1; ++j) {
									tmp_image[j+i*ny] = image[j+i*ny] * 0.6f + (image[j  +(i-1)*ny] + image[j  +(i+1)*ny] + image[j-1+i*ny] + image[j+1+i*ny]) * 0.1f;
							}
					}

					//bottom left cell
					tmp_image[(ny-1) * ny] = image[(ny-1) * ny] * 0.6f + (image[(ny-2) * ny] + image[1 + (ny-1) * ny]) * 0.1f;

					//bottom row
					for(int i = 1; i < ny - 1; ++i){
							tmp_image[(ny - 1) * ny + i] = image[(ny - 1) * ny + i] * 0.6f + (image[(ny - 1) * ny + (i - 1)] + image[(ny - 1) * ny + (i + 1)] + image[(ny - 2) * ny + i]) * 0.1f;
					}

					//bottom right cell
					tmp_image[(ny - 1) + (ny - 1) * ny] = image[(ny - 1) + (ny - 1) * ny] * 0.6f + (image[(ny - 2) + (ny - 1) * ny] + image[(ny - 1) + (ny - 2) * ny]) * 0.1f;
					//top left cell
					image[0] = tmp_image[0] * 0.6f + (tmp_image[ny] + tmp_image[1]) * 0.1f;

					//top row
					for(int i = 1; i < ny - 1; ++i){
							image[i] = tmp_image[i] * 0.6f + (tmp_image[i - 1] + tmp_image[i + 1] + tmp_image[ny + i]) * 0.1f;
					}
					//top right cell
					image[ny-1] = tmp_image[ny-1] * 0.6f + (tmp_image[ny-2] + tmp_image[2*  ny-1]) * 0.1f;
					//left side column
					for(int j = 1; j < ny - 1; ++j){
							image[ny * j] = tmp_image[ny * j] * 0.6f + (tmp_image[ny * (j - 1)] + tmp_image[ny * (j + 1)] + tmp_image[ny * j + 1]) * 0.1f;
					}
					//right side column
					for(int j = 1; j < ny - 1; ++j){
							image[ny * j + ny - 1] = tmp_image[ny * j + ny - 1] * 0.6f + (tmp_image[ny * (j - 1) + ny - 1] + tmp_image[ny * (j + 1) + ny - 1] + tmp_image[ny * j + ny - 2]) * 0.1f;
					}
					//inner grid
					for (int i = 1; i < ny - 1; ++i) {
							for (int j = 1; j < nx - 1; ++j) {
									image[j+i*ny] = tmp_image[j+i*ny] * 0.6f + (tmp_image[j  +(i-1)*ny] + tmp_image[j  +(i+1)*ny] + tmp_image[j-1+i*ny] + tmp_image[j+1+i*ny]) * 0.1f;
							}
					}

					//bottom left cell
					image[(ny-1) * ny] = tmp_image[(ny-1) * ny] * 0.6f + (tmp_image[(ny-2) * ny] + tmp_image[1 + (ny-1) * ny]) * 0.1f;

					//bottom row
					for(int i = 1; i < ny - 1; ++i){
							image[(ny - 1) * ny + i] = tmp_image[(ny - 1) * ny + i] * 0.6f + (tmp_image[(ny - 1) * ny + (i - 1)] + tmp_image[(ny - 1) * ny + (i + 1)] + tmp_image[(ny - 2) * ny + i]) * 0.1f;
					}

					//bottom right cell
					image[(ny - 1) + (ny - 1) * ny] = tmp_image[(ny - 1) + (ny - 1) * ny] * 0.6f + (tmp_image[(ny - 2) + (ny - 1) * ny] + tmp_image[(ny - 1) + (ny - 2) * ny]) * 0.1f;				}

		}

		toc = wtime();

		if (rank == MASTER) {
				printf ("%lf\n", toc-tic);
		}
		free(sendbuf);
		free(recvbuf);

		if (rank != MASTER && size != 1) {
				MPI_Ssend(image, nx * ny, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
		} else if (size != 1) {
				for (int i = 1; i < size; i++) {
						MPI_Recv(tmp_image, ny * nx, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
						if (i != size - 1) {
								for (int j = 0; j < (lastrow + 1) * nx + 1; j++){
										image[i * nx * local_nrows + j] = tmp_image[i * nx * local_nrows + j];
								}
						} else {
								for (int j = (lastrow + 1) * i * nx ; j < (nx) * (ny - 1) + nx; j++){
										image[j] = tmp_image[j];
								}
						}
				}
		}

		if (rank == MASTER) output_image(OUTPUT_FILE, nx, ny, image);
		free(image);
		free(tmp_image);
		MPI_Finalize();
		return EXIT_SUCCESS;

}

void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image, int firstrow, int lastrow, float * restrict sendbuf, float * restrict recvbuf, int above, int below, MPI_Status status, int rank, int size) {

		if (rank != size - 1) {
			 MPI_Ssend(&image[lastrow * nx], nx, MPI_FLOAT, below, tag, MPI_COMM_WORLD);
		}
		if (rank != MASTER) {
				MPI_Recv(&image[(firstrow - 1)* nx], nx, MPI_FLOAT, above, tag, MPI_COMM_WORLD, &status);
		}

	//	MPI_Sendrecv(&image[lastrow * nx], nx, MPI_FLOAT, below, tag, &image[(firstrow - 1)* nx], nx, MPI_FLOAT, above, tag, MPI_COMM_WORLD, &status);
		//if top section
		if (firstrow == 0) {
				//top left cell
				tmp_image[0] = image[0] * 0.6f + (image[nx] + image[1]) * 0.1f;

				//top row
				for(int i = 1; i < nx - 1; ++i){
						tmp_image[i] = image[i] * 0.6f + (image[i - 1] + image[i + 1] + image[nx + i]) * 0.1f;
				}

				//top right cell
				tmp_image[nx - 1] = image[nx - 1] * 0.6f + (image[nx - 2] + image[2 * nx - 1]) * 0.1f;

		//any other section
		} else {
				//top left
				tmp_image[firstrow * nx] = image[firstrow * nx] * 0.6f + (image[(firstrow + 1) * nx] + image[(firstrow * nx) + 1] + image[(firstrow - 1) * nx]) * 0.1f;
				//top row
				for(int i = 1; i < nx - 1; ++i){
						tmp_image[firstrow * nx + i] = image[firstrow * nx + i] * 0.6f + (image[firstrow * nx + i - 1] + image[firstrow * nx + i + 1] + image[(firstrow + 1)* nx + i] + image[(firstrow - 1) * nx + i]) * 0.1f;
				}
				//top right cell
				tmp_image[(firstrow + 1) * nx - 1] = image[(firstrow + 1) * nx - 1] * 0.6f + (image[(firstrow + 1) * nx - 2] + image[(firstrow + 2) * nx - 1] + image[(firstrow - 1) * nx + nx - 1]) * 0.1f;
		}
		//left side column
		for(int j = firstrow + 1; j < lastrow; ++j){
				tmp_image[j * nx] = image[j * nx] * 0.6f + (image[(j - 1) * nx] + image[(j + 1) * nx] + image[j * nx + 1]) * 0.1f;
		}

		//right side column
		for(int j = firstrow + 1; j < lastrow; ++j){
				tmp_image[(j + 1) * nx - 1] = image[(j + 1) * nx - 1] * 0.6f + (image[(j + 1) * nx - 2] + image[j * nx - 1] + image[(j + 2) * nx - 1]) * 0.1f;
		}

		//inner grid
		for (int i = firstrow + 1; i < lastrow; ++i) {
				for (int j = 1; j < nx - 1; ++j) {
				tmp_image[j+i*ny] = image[j+i*ny] * 0.6f + (image[j  +(i-1)*ny] + image[j  +(i+1)*ny] + image[j-1+i*ny] + image[j+1+i*ny]) * 0.1f;
				}
		}

		if (rank != MASTER) {
				MPI_Ssend(&image[firstrow * nx], nx, MPI_FLOAT, above, tag, MPI_COMM_WORLD);
		}
		if (rank != size - 1){
				MPI_Recv(&image[(lastrow + 1) * nx], nx, MPI_FLOAT, below, tag, MPI_COMM_WORLD, &status);
		}
	//	MPI_Sendrecv(&image[firstrow * nx], nx, MPI_FLOAT, above, tag,&image[(lastrow + 1) * nx], nx, MPI_FLOAT, below, tag, MPI_COMM_WORLD, &status);

		//if last section
		if (lastrow == ny - 1) {

				//bottom left cell
				tmp_image[(ny - 1) * nx] = image[(ny - 1) * nx] * 0.6f + (image[(ny - 2) * nx] + image[(ny - 1) * nx + 1]) * 0.1f;

				//bottom row
				for(int i = 1; i < ny - 1; ++i){
						tmp_image[(ny - 1) * nx + i] = image[(ny - 1) * nx + i] * 0.6f + (image[(ny - 1) * nx + i + 1] + image[(ny - 1) * nx + i - 1] + image[(ny - 2) * nx + i]) * 0.1f;
				}

				//bottom right cell
				tmp_image[nx - 1 + (ny - 1) * nx] = image[nx - 1 + (ny - 1) * nx] * 0.6f + (image[nx - 2 + (ny - 1) * nx] + image[nx - 1 + (ny - 2) * nx]) * 0.1f;

		//any other section
		} else {

				//bottom left
				tmp_image[lastrow * nx] = image[lastrow * nx] * 0.6f + (image[(lastrow - 1) * nx] + image[(lastrow * nx) + 1] + image[(lastrow + 1) * nx]) * 0.1f;

				//bottom row
						for(int i = 1; i < nx - 1; ++i){
								tmp_image[lastrow * nx + i] = image[lastrow * nx + i] * 0.6f + (image[lastrow * nx + i - 1] + image[lastrow * nx + i + 1] + image[(lastrow - 1)* nx + i] + image[(lastrow + 1) * nx + i]) * 0.1f;
						}

				//bottom right cell
				tmp_image[(lastrow + 1) * nx - 1] = image[(lastrow + 1) * nx - 1] * 0.6f + (image[(lastrow + 1) * nx - 2] + image[(lastrow) * nx - 1] + image[(lastrow + 1) * nx + nx - 1]) * 0.1f;
		}
}

// Create the input image
void init_image(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {

		//Init to 0
		for (int y = 0; y < ny; y++) {
				for (int x = 0; x < nx; x++) {
						image[y*ny+x] = 0.0f;
						tmp_image[y*ny+x] = 0.0f;
				}
		}

		//Init to checkboard
		//Init to checkboard
			for (int j = 0; j < 8; j++) {
					for (int i = 0; i < 8; i++) {
						 for (int jj = j*ny/8; jj < (j+1)*ny/8; jj++) {
								 for (int ii = i*nx/8; ii < (i+1)*nx/8; ii++) {
										 if((i+j)%2) image[jj+ii*ny] = 100.0f;
								}
						 }
				 }
			}
	}


int calc_nrows_from_rank(int rank, int size, int ny) {
		int nrows;
		nrows = ny/size;
		return nrows;
}

void output_image(const char * file_name, const int nx, const int ny, float * restrict image) {

	// Open output file
		FILE *fp = fopen(file_name, "w");
		if (!fp) {
				fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
				exit(EXIT_FAILURE);
		}

		// Ouptut image header
		fprintf(fp, "P5 %d %d 255\n", nx, ny);

		// Calculate maximum value of image
		// This is used to rescale the values
		// to a range of 0-255 for output
		float  maximum = 0.0f;
		for (int j = 0; j < ny; ++j) {
				for (int i = 0; i < nx; ++i) {
						if (image[i+j*nx] > maximum)
							maximum = image[i+j*nx];
				}
		}

		// Output image, converting to numbers 0-255
		for (int j = 0; j < ny; ++j) {
				for (int i = 0; i < nx; ++i) {
						fputc((char)(255.0*image[i+j*nx]/maximum), fp);
				}
		}

		// Close the file
		fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec*1e-6;
}
