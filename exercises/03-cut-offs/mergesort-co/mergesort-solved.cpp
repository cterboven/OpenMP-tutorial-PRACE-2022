/*
*  This file is part of Christian's OpenMP software lab 
*
*  Copyright (C) 2016 by Christian Terboven <terboven@itc.rwth-aachen.de>
*  Copyright (C) 2016 by Jonas Hahnfeld <hahnfeld@itc.rwth-aachen.de>
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/time.h>

#include <iostream>
#include <algorithm>

#include <cstdlib>
#include <cstdio>

#include <cmath>
#include <ctime>
#include <cstring>

#include <omp.h>


/**
  * helper routine: check if array is sorted correctly
  */
bool isSorted(int ref[], int data[], const size_t size){
	std::sort(ref, ref + size);
	for (size_t idx = 0; idx < size; ++idx){
		if (ref[idx] != data[idx]) {
			return false;
		}
	}
	return true;
}


/**
  * sequential merge step (straight-forward implementation)
  */
void MsMergeSequential(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin) {
	long left = begin1;
	long right = begin2;

	long idx = outBegin;

	while (left < end1 && right < end2) {
		if (in[left] <= in[right]) {
			out[idx] = in[left];
			left++;
		} else {
			out[idx] = in[right];
			right++;
		}
		idx++;
	}

	while (left < end1) {
		out[idx] = in[left];
		left++, idx++;
	}

	while (right < end2) {
		out[idx] = in[right];
		right++, idx++;
	}
}


/**
  * sequential MergeSort
  */
void MsSequential(int *array, int *tmp, bool inplace, long begin, long end) {
	if (begin < (end - 1)) {
		const long half = (begin + end) / 2;
		MsSequential(array, tmp, !inplace, begin, half);
		MsSequential(array, tmp, !inplace, half, end);
		if (inplace) {
			MsMergeSequential(array, tmp, begin, half, half, end, begin);
		} else {
			MsMergeSequential(tmp, array, begin, half, half, end, begin);
		}
	} else if (!inplace) {
		tmp[begin] = array[begin];
	}
}

/**
  * parallel merge step (straight-forward implementation)
  */
void MsMergeParallel(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin, int deep) {
	if (deep) {
		long half1, half2, tmp, count, step;
		if ((end1 - begin1) < (end2 - begin2)) {
			half2 = (begin2 + end2) / 2;
			// find in[half2] in [begin1, end1) (std::upper_bound)
			half1 = begin1, count = (end1 - begin1);
			while (count > 0) {
				step = count / 2;
				tmp = half1 + step;
				if (in[tmp] <= in[half2]) {
					tmp++;
					half1 = tmp;
					count -= step + 1;
				} else {
					count = step;
				}
			}
		} else {
			half1 = (begin1 + end1) / 2;
			// find in[half1] in [begin2, end2) (std::lower_bound)
			half2 = begin2, count = (end2 - begin2);
			while (count > 0) {
				step = count / 2;
				tmp = half2 + step;
				if (in[tmp] < in[half1]) {
					tmp++;
					half2 = tmp;
					count -= step + 1;
				} else {
					count = step;
				}
			}
		}

		#pragma omp task default(shared)
		{
			MsMergeParallel(out, in, begin1, half1, begin2, half2, outBegin, deep - 1);
		}

		long outBegin2 = outBegin + (half1 - begin1) + (half2 - begin2);
		#pragma omp task default(shared)
		{
			MsMergeParallel(out, in, half1, end1, half2, end2, outBegin2, deep - 1);
		}

		#pragma omp taskwait
	} else {
		MsMergeSequential(out, in, begin1, end1, begin2, end2, outBegin);
	}
}

/**
  * OpenMP Task-parallel MergeSort
  */
void MsParallel(int *array, int *tmp, bool inplace, long begin, long end, int deep) {
	if (begin < (end - 1)) {
		long half = (begin + end) / 2;
		if (deep){
			#pragma omp task default(shared)
			{
				MsParallel(array, tmp, !inplace, begin, half, deep - 1);
			}

			#pragma omp task default(shared)
			{
				MsParallel(array, tmp, !inplace, half, end, deep - 1);
			}

			#pragma omp taskwait
		}
		else {
			MsSequential(array, tmp, !inplace, begin, half);
			MsSequential(array, tmp, !inplace, half, end);
		}

		if (inplace) {
			MsMergeParallel(array, tmp, begin, half, half, end, begin, deep);
		} else {
			MsMergeParallel(tmp, array, begin, half, half, end, begin, deep);
		}
	} else if (!inplace) {
		tmp[begin] = array[begin];
	}
}


/**
  * OpenMP Task-parallel MergeSort
  * startup routine containing the Parallel Region
  */
void MsParallelOmp(int *array, int *tmp, const size_t size) {

	// compute cut-off recursion level
	const int iMinTask = (omp_get_max_threads() * 5);
	int deep = 0;
	while ((1 << deep) < iMinTask) deep += 1;

#pragma omp parallel
#pragma omp master
	{
		MsParallel(array, tmp, true, 0, size, deep);
	}
}

/** 
  * @brief program entry point
  */
int main(int argc, char* argv[]) {
	// variables to measure the elapsed time
	struct timeval t1, t2;
	double etime;

	// expect one command line arguments: array size
	if (argc != 2) {
		printf("Usage: MergeSort.exe <array size> \n");
		printf("\n");
		return EXIT_FAILURE;
	}
	else {
		const size_t stSize = strtol(argv[1], NULL, 10);
		int *data = (int*) malloc(stSize * sizeof(int));
		int *tmp = (int*) malloc(stSize * sizeof(int));
		int *ref = (int*) malloc(stSize * sizeof(int));

		// first touch
		#pragma omp parallel for
		for (size_t idx = 0; idx < stSize; ++idx){
			data[idx] = 0;
			tmp[idx] = 0;
		}

		printf("Initialization...\n");

		srand(95);
		for (size_t idx = 0; idx < stSize; ++idx){
			data[idx] = (int) (stSize * (double(rand()) / RAND_MAX));
		}
		std::copy(data, data + stSize, ref);

		double dSize = (stSize * sizeof(int)) / 1024 / 1024;
		printf("Sorting %zu elements of type int (%f MiB)...\n", stSize, dSize);

		gettimeofday(&t1, NULL);
		MsParallelOmp(data, tmp, stSize);
		gettimeofday(&t2, NULL);

		etime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
		etime = etime / 1000;

		printf("done, took %f sec. Verification...", etime);
		if (isSorted(ref, data, stSize)) {
			printf(" successful.\n");
		}
		else {
			printf(" FAILED.\n");
		}

		free(data);
		free(tmp);
		free(ref);
	}

	return EXIT_SUCCESS;
}
