// main.go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// ----------------- utils -----------------

// Функция активации — сигмоида.
// Она преобразует любое число в диапазон (0, 1).
// Используется для моделирования "активации нейрона".
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Производная сигмоиды.
// Нужна при обучении для расчёта градиентов во время обратного распространения ошибки.
func sigmoidPrime(x float64) float64 {
	s := sigmoid(x)
	return s * (1 - s)
}

// Возвращает случайное число в диапазоне [-1, 1].
// Используется для инициализации весов нейронов.
func randomFloat() float64 {
	return (rand.Float64()*2 - 1)
}

// ----------------- Layer -----------------

// Структура Layer описывает один слой нейронной сети.
type Layer struct {
	in      int           // Количество входов (нейронов предыдущего слоя)
	out     int           // Количество выходов (нейронов текущего слоя)
	weights [][]float64   // Матрица весов [out][in]
	biases  []float64     // Вектор смещений (bias) [out]
	zs      [][]float64   // Сырые суммы (до функции активации), нужны для обратного прохода
	as      [][]float64   // Активации (результат функции активации)
}

// Функция-конструктор нового слоя.
// Принимает количество входов и выходов, создаёт и случайно инициализирует веса и смещения.
func NewLayer(in, out int) *Layer {
	L := &Layer{
		in:      in,
		out:     out,
		weights: make([][]float64, out),
		biases:  make([]float64, out),
		zs:      make([][]float64, 0),
		as:      make([][]float64, 0),
	}
	// Инициализация весов и смещений случайными числами
	for i := 0; i < out; i++ {
		L.weights[i] = make([]float64, in)
		for j := 0; j < in; j++ {
			L.weights[i][j] = randomFloat()
		}
		L.biases[i] = randomFloat()
	}
	return L
}

// Прямое распространение (forward pass) для одного образца.
// На вход подаётся вектор входных данных, возвращается вектор выходных активаций.
func (L *Layer) Forward(x []float64) []float64 {
	z := make([]float64, L.out) // сумма входов * вес + смещение
	a := make([]float64, L.out) // результат после функции активации
	for i := 0; i < L.out; i++ {
		sum := L.biases[i] // начинаем с bias
		for j := 0; j < L.in; j++ {
			sum += L.weights[i][j] * x[j]
		}
		z[i] = sum
		a[i] = sigmoid(sum) // применяем сигмоиду
	}
	// Сохраняем значения для обратного распространения ошибки
	L.zs = [][]float64{z}
	L.as = [][]float64{a}
	return a
}

// ----------------- Network -----------------

// Структура Network описывает всю нейросеть (последовательность слоёв).
type Network struct {
	layers []*Layer
}

// Создание сети по списку размеров слоёв, например [2, 2, 1]:
// 2 входа → 2 нейрона в скрытом → 1 выход.
func NewNetwork(sizes []int) *Network {
	layers := make([]*Layer, 0)
	for i := 0; i < len(sizes)-1; i++ {
		layers = append(layers, NewLayer(sizes[i], sizes[i+1]))
	}
	return &Network{layers: layers}
}

// Прямое распространение сигнала через все слои сети.
func (net *Network) Feedforward(x []float64) []float64 {
	in := x
	for _, L := range net.layers {
		in = L.Forward(in)
	}
	return in
}

// ----------------- Обучение сети -----------------

// Функция Train обучает сеть методом стохастического градиентного спуска (SGD).
// samples — входные данные
// targets — правильные ответы
// epochs — количество эпох (итераций обучения)
// lr — learning rate (скорость обучения)
func (net *Network) Train(samples []([]float64), targets []([]float64), epochs int, lr float64) {
	if len(samples) != len(targets) {
		panic("Количество образцов и целей должно совпадать")
	}

	for e := 0; e < epochs; e++ {
		totalLoss := 0.0 // Для расчёта средней ошибки на эпоху

		// Проходим по всем обучающим образцам
		for idx, x := range samples {
			y := targets[idx]

			// ---------- Прямое распространение ----------
			activations := make([][]float64, 0) // Список всех активаций по слоям
			zValues := make([][]float64, 0)     // Список всех z = (w*x + b)
			in := x
			activations = append(activations, in) // Добавляем входной слой

			for _, L := range net.layers {
				a := L.Forward(in)
				zValues = append(zValues, L.zs[0])
				activations = append(activations, a)
				in = a
			}

			output := activations[len(activations)-1]

			// ---------- Расчёт ошибки ----------
			// Используем среднеквадратичную ошибку (MSE)
			for i := 0; i < len(y); i++ {
				diff := output[i] - y[i]
				totalLoss += 0.5 * diff * diff
			}

			// ---------- Обратное распространение ошибки (backpropagation) ----------

			// Массив дельт для каждого слоя
			deltas := make([][]float64, len(net.layers))

			// --- Выходной слой ---
			Lout := net.layers[len(net.layers)-1]
			deltaOut := make([]float64, Lout.out)
			for i := 0; i < Lout.out; i++ {
				a := activations[len(activations)-1][i]
				z := zValues[len(zValues)-1][i]
				// Формула: delta = (a - y) * f'(z)
				deltaOut[i] = (a - y[i]) * sigmoidPrime(z)
			}
			deltas[len(net.layers)-1] = deltaOut

			// --- Скрытые слои (обратное распространение дельт) ---
			for l := len(net.layers) - 2; l >= 0; l-- {
				L := net.layers[l]
				Lnext := net.layers[l+1]
				delta := make([]float64, L.out)

				for i := 0; i < L.out; i++ {
					sum := 0.0
					// Суммируем ошибку от нейронов следующего слоя
					for k := 0; k < Lnext.out; k++ {
						sum += Lnext.weights[k][i] * deltas[l+1][k]
					}
					z := zValues[l][i]
					// Формула: delta = (W_next^T * delta_next) * f'(z)
					delta[i] = sum * sigmoidPrime(z)
				}
				deltas[l] = delta
			}

			// ---------- Обновление весов и смещений ----------
			for l := 0; l < len(net.layers); l++ {
				L := net.layers[l]
				aPrev := activations[l] // активации предыдущего слоя

				for i := 0; i < L.out; i++ {
					// Обновляем bias (смещение)
					L.biases[i] -= lr * deltas[l][i]

					// Обновляем веса
					for j := 0; j < L.in; j++ {
						grad := deltas[l][i] * aPrev[j] // градиент dC/dW = delta * a_prev
						L.weights[i][j] -= lr * grad
					}
				}
			}
		}

		// Каждые 1000 эпох выводим среднюю ошибку
		if e%1000 == 0 || e == epochs-1 {
			avgLoss := totalLoss / float64(len(samples))
			fmt.Printf("Эпоха %d: средняя ошибка = %.6f\n", e, avgLoss)
		}
	}
}

// Функция предсказания — просто прямое прохождение сети без обучения.
func (net *Network) Predict(x []float64) []float64 {
	return net.Feedforward(x)
}

// ----------------- Пример использования (XOR) -----------------
func main() {
	rand.Seed(time.Now().UnixNano()) // Инициализация генератора случайных чисел

	// Пример данных для задачи XOR:
	// Выход равен 1, если только один из входов равен 1
	samples := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// Создаём сеть: 2 входа → 2 нейрона в скрытом → 1 выход
	net := NewNetwork([]int{2, 2, 1})

	// Параметры обучения
	epochs := 20000      // количество эпох (итераций)
	learningRate := 0.5  // скорость обучения

	// Запускаем обучение
	net.Train(samples, targets, epochs, learningRate)

	// Проверяем сеть после обучения
	fmt.Println("\nРезультаты после обучения:")
	for i, s := range samples {
		out := net.Predict(s)
		fmt.Printf("Вход: %v → Выход: %.4f (округлён %d), Цель: %v\n",
			s, out[0], int(math.Round(out[0])), targets[i])
	}
}
