#include <stdlib.h>
#include <stdio.h>

int main() {
    FILE *fpOrigem, *fpDestino;
    fpOrigem = fopen("bm.txt", "r");
    fpDestino = fopen("bm_final.txt", "w");
    
    if (!fpOrigem || !fpDestino){
        puts("Impossivel abrir o arquivo");
        return 1;
    }

    double custo, tempo, c, t;
    int i;
    char instancia[30];

    while(1){
        if (feof(fpOrigem))
            break;

        custo = 0;
        tempo = 0;

        fgets(instancia, 30, fpOrigem);
        
        for (i = 0; i < 10; i++){
            fscanf(fpOrigem, "%*s");
            fscanf(fpOrigem, "%lf", &c);
            fscanf(fpOrigem, "%*s");
            fscanf(fpOrigem, "%lf%*c", &t);
            
            custo += c;
            tempo += t;
        }

        fprintf(fpDestino, "%sCusto: %lf\nTempo: %lf\n\n", instancia, custo/10, tempo/10);
    }
    
    fclose(fpOrigem);
    fclose(fpDestino);

	return 0;
}