#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

typedef unsigned short mtype;

/* Lê uma sequência de um arquivo para um vetor de char.
   O nome do arquivo é passado como parâmetro. */
char* read_seq(char *fname) {
    FILE *fseq = NULL;
    long size = 0;
    char *seq = NULL;
    int i = 0;

    fseq = fopen(fname, "rt");
    if (fseq == NULL) {
        printf("Error reading file %s\n", fname);
        exit(1);
    }

    // Determina o tamanho da sequência para alocar memória
    fseek(fseq, 0L, SEEK_END);
    size = ftell(fseq);
    rewind(fseq);

    seq = (char *) calloc(size + 1, sizeof(char));
    if (seq == NULL) {
        printf("Erro allocating memory for sequence %s.\n", fname);
        exit(1);
    }

    while (!feof(fseq)) {
        int c = fgetc(fseq);
        if ((c != '\n') && (c != EOF))
            seq[i++] = (char)c;
    }
    seq[i] = '\0';

    fclose(fseq);
    return seq;
}

/*
 * Aloca a matriz de pontuação.
 * Aqui, a alocação do vetor de linhas é sequencial (precisa ser feito antes),
 * mas a alocação de cada linha é paralelizável.
 */
mtype ** allocateScoreMatrix(int sizeA, int sizeB) {
    int i;
    mtype **scoreMatrix = (mtype **) malloc((sizeB + 1) * sizeof(mtype *));
    if (scoreMatrix == NULL) {
        printf("Erro allocating memory for scoreMatrix pointer.\n");
        exit(1);
    }
    // Alocação paralela de cada linha da matriz
    #pragma omp parallel for default(none) private(i) shared(scoreMatrix, sizeA, sizeB)
    for (i = 0; i < (sizeB + 1); i++) {
        scoreMatrix[i] = (mtype *) malloc((sizeA + 1) * sizeof(mtype));
        if (scoreMatrix[i] == NULL) {
            printf("Erro allocating memory for row %d.\n", i);
            exit(1);
        }
    }
    return scoreMatrix;
}

/*
 * Inicializa a matriz: primeira linha e primeira coluna com zeros.
 */
void initScoreMatrix(mtype **scoreMatrix, int sizeA, int sizeB) {
    int i, j;
    // Paraleliza a inicialização da primeira linha
    #pragma omp parallel for default(none) shared(scoreMatrix, sizeA) private(j)
    for (j = 0; j < (sizeA + 1); j++) {
        scoreMatrix[0][j] = 0;
    }

    // Paraleliza a inicialização da primeira coluna
    #pragma omp parallel for default(none) shared(scoreMatrix, sizeB) private(i)
    for (i = 1; i < (sizeB + 1); i++) {
        scoreMatrix[i][0] = 0;
    }
}

/*
 * Cálculo paralelo da LCS utilizando abordagem de "wavefront".
 * A matriz é processada diagonalmente, onde cada diagonal pode ser paralelizada.
 */
int LCS(mtype **scoreMatrix, int sizeA, int sizeB, char *seqA, char *seqB) {
    int diag;
    // Itera pelas diagonais, onde diagonal varia de 2 até sizeA+sizeB
    for (diag = 2; diag <= sizeA + sizeB; diag++) {
        int i_start = (diag - sizeA > 1 ? diag - sizeA : 1);
        int i_end   = min(diag - 1, sizeB);
        
        #pragma omp parallel for default(none) shared(scoreMatrix, seqA, seqB, diag, i_start, i_end)
        for (int i = i_start; i <= i_end; i++) {
            int j = diag - i;
            if (seqA[j - 1] == seqB[i - 1])
                scoreMatrix[i][j] = scoreMatrix[i - 1][j - 1] + 1;
            else
                scoreMatrix[i][j] = max(scoreMatrix[i - 1][j], scoreMatrix[i][j - 1]);
        }
    }
    return scoreMatrix[sizeB][sizeA];
}

/*
 * (Opcional) Imprime a matriz de pontuação para fins de depuração.
 */
void printMatrix(char *seqA, char *seqB, mtype **scoreMatrix, int sizeA, int sizeB) {
    int i, j;
    printf("Score Matrix:\n");
    printf("========================================\n");

    printf("    ");
    printf("%5c   ", ' ');
    for (j = 0; j < sizeA; j++)
        printf("%5c   ", seqA[j]);
    printf("\n");

    for (i = 0; i < sizeB + 1; i++) {
        if (i == 0)
            printf("    ");
        else
            printf("%c   ", seqB[i - 1]);
        for (j = 0; j < sizeA + 1; j++) {
            printf("%5d   ", scoreMatrix[i][j]);
        }
        printf("\n");
    }
    printf("========================================\n");
}

/*
 * Libera a memória da matriz de pontuação.
 * Cada linha é liberada em paralelo.
 */
void freeScoreMatrix(mtype **scoreMatrix, int sizeB) {
    int i;
    #pragma omp parallel for default(none) shared(scoreMatrix, sizeB) private(i)
    for (i = 0; i < (sizeB + 1); i++) {
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