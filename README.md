# Задача Дирихле для уравнения Пуассона в криволинейной области

Проект по курсу "Суперкомпьютерное моделирование и технологии"


## Описание

Решение двумерной задачи Дирихле для уравнения Пуассона в криволинейной области (трапеция с вершинами A(-3,0), B(3,0), C(2,3), D(-2,3)) методом фиктивных областей с использованием метода сопряженных градиентов.

## Структура репозитория

### Исходные файлы

- `main_seq.cpp` - последовательная реализация
- `main_openmp.cpp` - реализация с использованием OpenMP
- `main_mpi.cpp` - реализация с использованием MPI
- `main_hybrid.cpp` - гибридная MPI+OpenMP реализация
- `main_mpi_cuda.cpp` - реализация MPI+CUDA
- `Makefile` - файл сборки для MPI+CUDA версии

### Отчет

- `main.pdf` - отчет
- `solution_800_1200_8_omp_*.jpg` - визуализация результатов

### Скрипты запуска

- `OpenMP_openmp.lsf` - скрипт для запуска OpenMP версии на кластере IBM Polus
- `OpenMP_hybrid.lsf` - скрипт для запуска гибридной MPI+OpenMP версии на кластере IBM Polus

## Компиляция

### Последовательная версия

```bash
g++ -O3 -o main_seq main_seq.cpp
```

### OpenMP версия

```bash
g++ -fopenmp -O3 -o main_openmp main_openmp.cpp
```

### MPI версия

```bash
module load SpectrumMPI
module load OpenMPI/4.0.2
mpic++ -O3 -o main_mpi main_mpi.cpp
```

### Гибридная MPI+OpenMP версия

```bash
module load SpectrumMPI
module load OpenMPI/4.0.2
mpic++ -fopenmp -O3 -o main_hybrid main_hybrid.cpp
```

### MPI+CUDA версия

```bash
module load SpectrumMPI
module load OpenMPI/4.0.2
make ARCH=sm_60 HOST_COMP=mpicc  # для GPU с compute capability 6.0
# или
make ARCH=sm_35 HOST_COMP=mpicc  # для GPU с compute capability 3.5
```

## Запуск

Последовательная версия:
```bash
./main_seq
```

OpenMP версия:
```bash
bsub < OpenMP_openmp.lsf
```


MPI версия:
```bash
mpisubmit.pl -p 16 -w 00:05 ./main_mpi -- 400 600
```

Гибридная версия (MPI+OpenMP):
```bash
bsub < OpenMP_hybrid.lsf
```


### MPI+CUDA версия

```bash
mpisubmit.pl -p 2 --gpu 2 -w 00:05 ./main_mpi_cuda -- 2400 3600
```
## Структура Git-репозитория

### Ветки репозитория:

- **`main`** - основная ветка с финальными версиями всех реализаций и готовым отчетом (`main.pdf`)
  - Содержит: `main_seq.cpp`, `main_openmp.cpp`, `main_mpi.cpp`, `main_hybrid.cpp`, `main_mpi_cuda.cpp`, `Makefile`, `main.pdf`, скрипты запуска (`OpenMP_openmp.lsf`, `OpenMP_hybrid.lsf`)

- **`sequential`** - последовательная реализация
  - Содержит: `main_seq.cpp`

- **`openmp`** - OpenMP реализация
  - Содержит: `main_openmp.cpp`

- **`mpi`** - MPI реализация
  - Содержит: `main_mpi.cpp`

- **`hybrid`** - гибридная MPI+OpenMP реализация
  - Содержит: `main_hybrid.cpp`

- **`mpi-cuda`** - MPI+CUDA реализация
  - Содержит: `main_mpi_cuda.cpp`, `Makefile`




## Результаты

Детальные результаты измерений производительности представлены в отчете `main.pdf`.

