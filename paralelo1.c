#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

typedef unsigned short mtype;

/* Função para ler a sequência de um arquivo */
char* read_seq(char *fname) {
    FILE *fseq = fopen(fname, "rt");
    if (fseq == NULL) {
        printf("Error reading file %s\n", fname);
        exit(1);
    }
    fseek(fseq, 0L, SEEK_END);
    long size = ftell(fseq);
    rewind(fseq);

    char *seq = (char *) calloc(size + 1, sizeof(char));
    if (seq == NULL) {
        printf("Error allocating memory for sequence %s.\n", fname);
        exit(1);
    }

    int i = 0;
    while (!feof(fseq)) {
        int c = fgetc(fseq);
        if ((c != '\n') && (c != EOF))
            seq[i++] = (char)c;
    }
    seq[i] = '\0';
    fclose(fseq);
    return seq;
}

/* Paraleliza a alocação das linhas da matriz de pontuação */
mtype ** allocateScoreMatrix(int sizeA, int sizeB) {
    int i;
    mtype ** scoreMatrix = (mtype **) malloc((sizeB + 1) * sizeof(mtype *));
    #pragma omp parallel for shared(scoreMatrix, sizeA, sizeB)
    for (i = 0; i < (sizeB + 1); i++) {
        scoreMatrix[i] = (mtype *) malloc((sizeA + 1) * sizeof(mtype));
    }
    return scoreMatrix;
}

/* Inicializa a matriz: a primeira linha e a primeira coluna são zeradas */
void initScoreMatrix(mtype ** scoreMatrix, int sizeA, int sizeB) {
    int j;
    #pragma omp parallel for shared(scoreMatrix, sizeA, sizeB)
    for (j = 0; j < (sizeA + 1); j++) {
        scoreMatrix[0][j] = 0;
    }
    int i;
    #pragma omp parallel for shared(scoreMatrix, sizeA, sizeB)
    for (i = 1; i < (sizeB + 1); i++) {
        scoreMatrix[i][0] = 0;
    }
}

/* Computa o LCS usando uma varredura em anti-diagonais para extrair paralelismo */
int LCS(mtype ** scoreMatrix, int sizeA, int sizeB, char * seqA, char * seqB) {
    int d, i, j;
    int start, end;
    /* 
       Nota:
       Cada célula (i,j) depende de scoreMatrix[i-1][j], scoreMatrix[i][j-1] 
       e scoreMatrix[i-1][j-1]. Ao percorrer a matriz em anti-diagonais, as dependências 
       das células de uma diagonal já foram calculadas na iteração anterior.
    */
    #pragma omp parallel private(d, i, j, start, end)
    {
        for (d = 2; d <= (sizeA + sizeB); d++) {
            /* 
               Para cada diagonal d = i + j, i varia de max(1, d-sizeA) até min(sizeB, d-1)
               e j = d - i.
            */
            start = (d - sizeA > 1) ? d - sizeA : 1;
            end = (d - 1 > sizeB) ? sizeB : d - 1;
            #pragma omp for
            for (i = start; i <= end; i++) {
                j = d - i;
                if (seqA[j - 1] == seqB[i - 1])
                    scoreMatrix[i][j] = scoreMatrix[i - 1][j - 1] + 1;
                else
                    scoreMatrix[i][j] = max(scoreMatrix[i - 1][j], scoreMatrix[i][j - 1]);
            }
            #pragma omp barrier
        }
    }
    return scoreMatrix[sizeB][sizeA];
}

/* Função para impressão da matriz (útil para debug) */
void printMatrix(char * seqA, char * seqB, mtype ** scoreMatrix, int sizeA, int sizeB) {
    int i, j;
    printf("Score Matrix:\n");
    printf("========================================\n");
    printf("    ");
    printf("%5c   ", ' ');
    for (j = 0; j < sizeA; j++)
        printf("%5c   ", seqA[j]);
    printf("\n");
    for (i = 0; i < (sizeB + 1); i++) {
        if (i == 0)
            printf("    ");
        else
            printf("%c   ", seqB[i - 1]);
        for (j = 0; j < (sizeA + 1); j++)
            printf("%5d   ", scoreMatrix[i][j]);
        printf("\n");
    }
    printf("========================================\n");
}

/* Libera a memória alocada para a matriz de pontuação */
void freeScoreMatrix(mtype **scoreMatrix, int sizeB) {
    int i;
    #pragma omp parallel for shared(scoreMatrix, sizeA, sizeB)
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
    
    // medidas de tempo 
    double start, end, time_spent;

	//read both sequences
    start = omp_get_wtime();

	seqA = read_seq("fileA.in");
	seqB = read_seq("fileB.in");

	//find out sizes
	sizeA = strlen(seqA);
	sizeB = strlen(seqB);

    end = omp_get_wtime();
    time_spent = end - start;
    printf("Tempo de leitura das sequencias: %.6f segundos\n", time_spent);


	// allocate LCS score matrix
    start = omp_get_wtime();

	mtype ** scoreMatrix = allocateScoreMatrix(sizeA, sizeB);

    end = omp_get_wtime();
    time_spent = end - start;
    printf("Tempo de alocacao de memoria para matriz: %.6f segundos\n", time_spent);

	//initialize LCS score matrix
    start = omp_get_wtime();

	initScoreMatrix(scoreMatrix, sizeA, sizeB);

    end = omp_get_wtime();
    time_spent = end - start;
    printf("Tempo de inicializacao da matriz score: %.6f segundos\n", time_spent);

	//fill up the rest of the matrix and return final score (element locate at the last line and collumn)
    start = omp_get_wtime();

	mtype score = LCS(scoreMatrix, sizeA, sizeB, seqA, seqB);

    end = omp_get_wtime();
    time_spent = end - start;
    printf("Tempo de aplicacao do LCS: %.6f segundos\n", time_spent);

	/* if you wish to see the entire score matrix,
	 for debug purposes, define DEBUGMATRIX. */
#ifdef DEBUGMATRIX
	printMatrix(seqA, seqB, scoreMatrix, sizeA, sizeB);
#endif

	//print score
	printf("\nScore: %d\n", score);

	//free score matrix
    start = omp_get_wtime();

	freeScoreMatrix(scoreMatrix, sizeB);

    end = omp_get_wtime();
    time_spent = end - start;
    printf("Tempo de librecao da matriz: %.6f segundos\n", time_spent);

	return EXIT_SUCCESS;
}