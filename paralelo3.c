#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

typedef unsigned short mtype;

char* read_seq(const char *fname) {
	FILE *fseq = fopen(fname, "rt");
	if (fseq == NULL) {
		fprintf(stderr, "Error reading file %s\n", fname);
		exit(1);
	}

	fseek(fseq, 0L, SEEK_END);
	long size = ftell(fseq);
	rewind(fseq);

	char *seq = (char *)calloc(size + 1, sizeof(char));
	if (seq == NULL) {
		fprintf(stderr, "Error allocating memory for sequence %s.\n", fname);
		exit(1);
	}

	int i = 0;
	while (!feof(fseq)) {
		seq[i] = fgetc(fseq);
		if (seq[i] != '\n' && seq[i] != EOF)
			i++;
	}
	seq[i] = '\0';
	fclose(fseq);
	return seq;
}

mtype **allocateScoreMatrix(int sizeA, int sizeB) {
	mtype **scoreMatrix = (mtype **)malloc((sizeB + 1) * sizeof(mtype *));
	if (!scoreMatrix) {
		fprintf(stderr, "Erro alocando scoreMatrix\n");
		exit(1);
	}

	#pragma omp parallel for
	for (int i = 0; i <= sizeB; i++) {
		scoreMatrix[i] = (mtype *)malloc((sizeA + 1) * sizeof(mtype));
		if (!scoreMatrix[i]) {
			fprintf(stderr, "Erro alocando linha %d da scoreMatrix\n", i);
			exit(1);
		}
	}
	return scoreMatrix;
}

void initScoreMatrix(mtype **scoreMatrix, int sizeA, int sizeB) {
	#pragma omp parallel for
	for (int j = 0; j <= sizeA; j++)
		scoreMatrix[0][j] = 0;

	#pragma omp parallel for
	for (int i = 1; i <= sizeB; i++)
		scoreMatrix[i][0] = 0;
}

int LCS(mtype **scoreMatrix, int sizeA, int sizeB, char *seqA, char *seqB) {
	#pragma omp parallel
	{
		#pragma omp single
		for (int diag = 2; diag <= sizeA + sizeB; diag++) {
			int start_i = (diag <= sizeB + 1) ? diag - 1 : sizeB;
			int end_i = (diag - sizeA > 1) ? sizeA : diag - 1;

			for (int i = start_i; i >= (diag - end_i); i--) {
				int j = diag - i;

				if (i > 0 && i <= sizeB && j > 0 && j <= sizeA) {
					#pragma omp task firstprivate(i, j)
					{
						if (seqA[j - 1] == seqB[i - 1]) {
							scoreMatrix[i][j] = scoreMatrix[i - 1][j - 1] + 1;
						} else {
							scoreMatrix[i][j] = max(scoreMatrix[i - 1][j], scoreMatrix[i][j - 1]);
						}
					}
				}
			}
			#pragma omp taskwait
		}
	}

	return scoreMatrix[sizeB][sizeA];
}

void freeScoreMatrix(mtype **scoreMatrix, int sizeB) {
	#pragma omp parallel for
	for (int i = 0; i <= sizeB; i++) {
		free(scoreMatrix[i]);
	}
	free(scoreMatrix);
}

int main(int argc, char ** argv) {
	// sequence pointers for both sequences
	char *seqA, *seqB;

	// sizes of both sequences
	int sizeA, sizeB;

	//read both sequences
	seqA = read_seq("fileA.in");
	seqB = read_seq("fileB.in");

	//find out sizes
	sizeA = strlen(seqA);
	sizeB = strlen(seqB);

	// allocate LCS score matrix
	mtype ** scoreMatrix = allocateScoreMatrix(sizeA, sizeB);

	//initialize LCS score matrix
	initScoreMatrix(scoreMatrix, sizeA, sizeB);

	//fill up the rest of the matrix and return final score (element locate at the last line and collumn)
	mtype score = LCS(scoreMatrix, sizeA, sizeB, seqA, seqB);

	/* if you wish to see the entire score matrix,
	 for debug purposes, define DEBUGMATRIX. */
#ifdef DEBUGMATRIX
	printMatrix(seqA, seqB, scoreMatrix, sizeA, sizeB);
#endif

	//print score
	printf("\nScore: %d\n", score);

	//free score matrix
	freeScoreMatrix(scoreMatrix, sizeB);

	return EXIT_SUCCESS;
}
