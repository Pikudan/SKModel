# Задача Дирихле для уравнения Пуассона в криволинейной области

Проект по курсу "Суперкомпьютерное моделирование и технологии"

**Студент:** Пикуров Даниил  
**Группа:** № 617

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

- `main.tex` - исходный код отчета в LaTeX
- `main.pdf` - скомпилированный отчет
- `logo.eps` - логотип для титульной страницы
- `solution_800_1200_8_omp_*.jpg` - визуализация результатов

### Скрипты запуска

- `scripts/run_hybrid_*.lsf` - скрипты для запуска гибридной версии на кластере IBM Polus
- `scripts/run_mpi_cuda_*.lsf` - скрипты для запуска MPI+CUDA версии на кластере IBM Polus

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
module load SpectrumMPI  # или OpenMPI/4.0.2
mpic++ -O3 -o main_mpi main_mpi.cpp
```

### Гибридная MPI+OpenMP версия

```bash
module load SpectrumMPI
module load OpenMPI/4.0.2  # если нужно
mpic++ -fopenmp -O3 -o main_hybrid main_hybrid.cpp
```

### MPI+CUDA версия

```bash
module load SpectrumMPI
module load CUDA
make ARCH=sm_60 HOST_COMP=mpicc  # для GPU с compute capability 6.0
# или
make ARCH=sm_35 HOST_COMP=mpicc  # для GPU с compute capability 3.5
```

## Запуск

### Последовательная версия

```bash
./main_seq M N
```

где `M` и `N` - размеры сетки по осям x и y соответственно.

### OpenMP версия

```bash
export OMP_NUM_THREADS=4
./main_openmp M N
```

### MPI версия

```bash
mpirun -np 4 ./main_mpi M N
```

### Гибридная MPI+OpenMP версия

```bash
export OMP_NUM_THREADS=4
mpirun -np 2 ./main_hybrid M N 4
```

где последний аргумент - число потоков OpenMP на процесс.

### MPI+CUDA версия

```bash
mpirun -np 2 ./main_mpi_cuda M N
```

## Запуск на кластере IBM Polus

Используйте LSF скрипты для запуска на кластере:

```bash
bsub < scripts/run_hybrid_4proc_4threads_800_1200.lsf
bsub < scripts/run_mpi_cuda_2gpu_2400_3600.lsf
```

## Структура Git-репозитория

Проект организован с использованием веток Git:

- `main` - основная ветка с финальными версиями всех реализаций и отчетом
- `sequential` - последовательная реализация
- `openmp` - OpenMP реализация
- `mpi` - MPI реализация
- `hybrid` - гибридная MPI+OpenMP реализация
- `mpi-cuda` - MPI+CUDA реализация

История коммитов отражает процесс разработки и оптимизации на каждом этапе.

## Результаты

Детальные результаты измерений производительности представлены в отчете `main.pdf`.

### Примеры результатов для сетки 2400×3600:

- Последовательная программа: 2939.49 с
- OpenMP (20 потоков): 431.988 с (ускорение 6.81)
- MPI+OpenMP (20 процессов, 8 потоков): 255.755 с (ускорение 11.49)
- MPI+CUDA (1 процесс): 822.745 с (ускорение 3.57)
- MPI+CUDA (2 процесса): 419.607 с (ускорение 7.01)

## Требования

- Компилятор C++ с поддержкой C++11
- OpenMP (для OpenMP и гибридной версий)
- MPI (для MPI, гибридной и MPI+CUDA версий)
- CUDA Toolkit (для MPI+CUDA версии)
- GPU с compute capability >= 3.5 (для MPI+CUDA версии)

## Лицензия

Учебный проект.

