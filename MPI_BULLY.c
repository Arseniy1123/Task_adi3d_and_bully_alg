#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define TAG_ELECTION 1
#define TAG_OK 2
#define TAG_COORDINATOR 3

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int coordinator = -1;  // номер текущего координатора
    int election_in_progress = 1;  // флаг, показывающий, идут ли выборы

    while (1) {
        MPI_Status status;

        if (election_in_progress) {
            MPI_Send(NULL, 0, MPI_INT, rank - 1, TAG_OK, MPI_COMM_WORLD);

            // Процесс начинает выборы
            printf("Process %d initiates elections.\n", rank);
            fflush(stdout);
            for (int i = rank + 1; i < size; ++i) {
                MPI_Send(NULL, 0, MPI_INT, i, TAG_ELECTION, MPI_COMM_WORLD);
                printf("Process %d sent ВЫБОРЫ to process %d.\n", rank, i);
                fflush(stdout);
            }

            // Ожидаем ответов
            int noResponse = 1;
            for (int i = rank+1; i < size; ++i) {
                MPI_Request request;
                MPI_Isend(NULL, 0, MPI_INT, i, TAG_ELECTION, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
                MPI_Iprobe(i, TAG_OK, MPI_COMM_WORLD, &noResponse, &status);
                if (!noResponse) {
                    // Одному из процессов с большим номером пришло сообщение "ОК"
                    // Он берет на себя проведение выборов
                    election_in_progress = 0;
                    break;
                }
            }

            // Если нет ни одного ответа, P считается победителем
            if (noResponse) {
                coordinator = rank;
                printf("Process %d becomes coordinator.\n", rank);
                fflush(stdout);
                election_in_progress = 0;

                // Отправить сообщение "Координатор" всем процессам
                for (int i = 0; i < rank; ++i) {
                    MPI_Request request;
                    MPI_Isend(NULL, 0, MPI_INT, i, TAG_COORDINATOR, MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, MPI_STATUS_IGNORE);
                    printf("Process %d sent КООРДИНАТОР to process %d.\n", rank, i);
                    fflush(stdout);
                }                
                
                for (int i = rank+1; i < size; ++i) {
                    MPI_Request request;
                    MPI_Isend(NULL, 0, MPI_INT, i, TAG_COORDINATOR, MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, MPI_STATUS_IGNORE);
                    printf("Process %d sent КООРДИНАТОР to process %d.\n", rank, i);
                    fflush(stdout);
                }
            }
        }

        // Прием сообщений
        MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // Обработка сообщений
        if (status.MPI_TAG == TAG_ELECTION) {
            printf("Process %d received ВЫБОРЫ from process %d.\n", rank, status.MPI_SOURCE);
            fflush(stdout);
            // Начать выборы
            election_in_progress = 1;
        } else if (status.MPI_TAG == TAG_OK) {
            printf("Process %d received OK from process %d.\n", rank, status.MPI_SOURCE);
            fflush(stdout);
            // Завершить выборы
            election_in_progress = 0;
        } else if (status.MPI_TAG == TAG_COORDINATOR) {
            // Обновить координатора
            coordinator = status.MPI_SOURCE;
            election_in_progress = 0;
        }
    }

    MPI_Finalize();
    return 0;
}