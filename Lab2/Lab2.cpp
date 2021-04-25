//g++ Lab2.cpp -o Lab2 -lpthread
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <pthread.h>
#include <ctime>
#define NUM_THREADS 4 //Number of pthreads
#define _VALUE 0.504066 //Exact integral meaning
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

using namespace std;

double VALUE = 0;

struct thread_data {
	double start = 0.0;
	double end = 0.0;
	double eps = 0.0;
	double step = 0.0;
	int id = 0;
};

double func(double x) {
	return sin(1/x);
}

double trapezoidal(double n, double a, double b)
{
    // Grid spacing
    double h = (b-a)/n;

    // Computing sum of first and last terms
    // in above formula
    double s = func(a)+func(b);

    // Adding middle terms in above formula
    for (int i = 1; i < n; i++)
        s += 2*func(a+i*h);

    // h/2 indicates (b-a)/2n. Multiplying h/2
    // with s.
    return (h/2)*s;
}

double* balanced(double a, double b, int thread_num) {
	double a_inv = 1 / a;
	double b_inv = 1 / b;
	double linv = a_inv - b_inv;
	double part_linv = linv / thread_num;

	double* points = (double*) calloc (thread_num, sizeof(double));
	cout << "Points initial: \n";
	for(int i = 0; i <= thread_num; i++)
	{
		points[i] = 1 / (a_inv - part_linv * i);
		cout << points[i] << " ";
	}
	cout << "\n";

	return points;
}
double* balanced_inside(double a, double b, double eps, double* step, int thread_num) {
	(*step) = (b-a)/10;
	while(fabs(func(a)-func(a + (*step))) >= eps)
	{
		(*step) /= 10;
	}
	double* points = (double*) calloc (thread_num, sizeof(double));
	cout << "Points: \n";
	for(int i = 0; i <= thread_num; i++)
	{
		points[i] = a + i * (b - a)/thread_num;
		cout << points[i] << " ";
	}
	cout << "\n";

	return points;
}

double count_integral(double start, double end, double eps, double step) {
	int n = (end - start)/step;
	// while(fabs(trapezoidal(2 * (*n), start, end) - trapezoidal((*n), start, end)) >= eps)
	// 	(*n) *= 2;
	return trapezoidal(2*n, start, end);
}


void* count_thread(void* args) {
	time_t time = clock();
	struct thread_data* data = (struct thread_data*)args;

	double integral = count_integral(data -> start, data -> end, data -> eps / NUM_THREADS / NUM_THREADS , data -> step);
	VALUE += integral;
    // here should be lock
    pthread_mutex_lock(&mutex);
	cout << "Ellapsed time of " <<  pthread_self() << " : "  << (clock() - time) << "ms " \
			 << "Points num: " << (data -> end-data -> start)/(data -> step)  << " [" << data -> start <<  ";" << data -> end << "] " << integral << endl;
    pthread_mutex_unlock(&mutex);
}


int main(int argc, char** argv) {
	double eps = 0.0, a = 0.001, b = 1.0;
  if (argc < 2)
		eps = 0.001;
  else
		eps = atof(argv[1]);

	double* points = balanced(a, b, NUM_THREADS);

	for (int i = 0; i < NUM_THREADS; i++)
	{
		double step = 0;
		double* points_inside = balanced_inside(points[i], points[i + 1], eps, &step,  NUM_THREADS);
		pthread_t threads[NUM_THREADS];
		struct thread_data td[NUM_THREADS];

		for(int j = 0; j < NUM_THREADS; j++) {
					td[j] = {points_inside[j], points_inside[j + 1], eps, step, j};
			pthread_create(&threads[j], NULL, count_thread, (void*)&td[j]);

		}
		for(int j = 0; j < NUM_THREADS; j++) pthread_join(threads[j],NULL);
		free(points_inside);
	}
	free(points);
	cout << "integration value = " << VALUE << endl;
	cout << "real value - counted value = " << fabs(_VALUE - VALUE) << " eps = " << eps << endl;
	pthread_exit(NULL);
	return 0;
}
