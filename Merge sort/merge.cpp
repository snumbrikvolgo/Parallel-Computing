#include <stdio.h>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <time.h>

void  showVector(char* file, int *v, int n);
int * merge(int *A, int asize, int *B, int bsize);
void  swap(int *v, int i, int j);
void  m_sort(int *A, int min, int max);

double startT,stopT;
double startTime;

void  showVector(char* file, int *v, int n)
{
  std::ofstream myfile;
  myfile.open (file);
	for(int i = 0; i < n; i++)
	 myfile << v[i] << std::endl;
  myfile.close();
}

int * merge(int *A, int asize, int *B, int bsize) {
	int ai, bi, ci, i;
	int *C;
	int csize = asize+bsize;

	ai = 0;
	bi = 0;
	ci = 0;

	C = (int *)malloc(csize*sizeof(int));
	while ((ai < asize) && (bi < bsize)) {
		if (A[ai] <= B[bi]) {
			C[ci] = A[ai];
			ci++; ai++;
		} else {
			C[ci] = B[bi];
			ci++; bi++;
		}
	}

	if (ai >= asize)
		for (i = ci; i < csize; i++, bi++)
			C[i] = B[bi];
	else if (bi >= bsize)
		for (i = ci; i < csize; i++, ai++)
			C[i] = A[ai];

	for (i = 0; i < asize; i++)
		A[i] = C[i];
	for (i = 0; i < bsize; i++)
		B[i] = C[asize+i];

	return C;
}

void  swap(int *v, int i, int j)
{
	int t;
	t = v[i];
	v[i] = v[j];
	v[j] = t;
}

void  m_sort(int *A, int min, int max)
{
	int mrank  = (min+max)/2;
	int lowerCount = mrank  - min + 1;
	int upperCount = max - mrank ;

	if (max == min) {
		return;
	}
  else {
		m_sort(A, min, mrank );
		m_sort(A, mrank +1, max);
		merge(A + min, lowerCount, A + mrank  + 1, upperCount);
	}
}

int main(int argc, char **argv)
{
	int* data;
	int* chunk;
	int* other;
  int* chunk1;
	int m = 0;
	int rank, size;
	int whole = 0;
	int step = 0;
	MPI_Status status;

  int n = atoi(argv[1]);
  char* in_file = argv[2];
  // char* out_file = argv[3];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	startT = clock();

	if(rank == 0)
	{
		int remnant;
		srandom(clock());
		whole = n / size;
		remnant = n % size;

    FILE * in;
    in = fopen(in_file, "r");
		data = (int *)malloc((n + whole - remnant) * sizeof(int));
		for(int i = 0; i < n; i++)
      fscanf(in, "%d", &data[i]);
			// data[i] = random();
		if(remnant != 0)
		{
			for(int i = n; i < n + whole - remnant; i++)
				data[i] = 0;
			whole++;
		}
    fclose(in);

    //showVector(in_file, data, n);


		MPI_Bcast(&whole, 1, MPI_INT, 0, MPI_COMM_WORLD);
		chunk = (int *)malloc(whole * sizeof(int));
		MPI_Scatter(data, whole, MPI_INT, chunk, whole, MPI_INT, 0, MPI_COMM_WORLD);
    free(data);
		m_sort(chunk, 0, whole - 1);
	}
	else
	{
    MPI_Bcast(&whole, 1, MPI_INT, 0, MPI_COMM_WORLD);
		chunk = (int *)malloc(whole * sizeof(int));
		MPI_Scatter(data, whole, MPI_INT, chunk, whole, MPI_INT, 0, MPI_COMM_WORLD);
		m_sort(chunk, 0, whole - 1);
	}

	step = 1;
	while(step < size)
	{
		if(rank % (2 * step) == 0)
		{
			if(rank + step < size)
			{
				MPI_Recv(&m, 1, MPI_INT, rank +step, 0, MPI_COMM_WORLD, &status);
				other = (int *)malloc(m * sizeof(int));
				MPI_Recv(other, m, MPI_INT, rank + step, 0, MPI_COMM_WORLD, &status);
				chunk1 = merge(chunk, whole, other, m);
        free(chunk);
        free(other);
        chunk = chunk1;
				whole = whole + m;
			}
		}
		else
		{
			int near = rank - step;
			MPI_Send(&whole ,1,MPI_INT, near, 0,MPI_COMM_WORLD);
			MPI_Send(chunk, whole, MPI_INT, near, 0,MPI_COMM_WORLD);
      free(chunk);
			break;
		}
		step = step*2;
	}

	stopT = clock();
	if(rank == 0)
	{
		FILE * fout;
    FILE * res;
		res = fopen("result.txt", "a+");
		fprintf(res, "%d %d %f\n", n, size, (stopT-startT)/CLOCKS_PER_SEC);
    fclose(res);
		// fout = fopen(out_file,"w");
		// for(int i = 0; i < whole; i++)
		// 	fprintf(fout,"%d\n",chunk[i]);
		// fclose(fout);
    free(chunk);

	}
	MPI_Finalize();
}
