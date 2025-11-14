package nn

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

// -----------------------------
// Сеть (Network)
// -----------------------------
type Network struct {
	Layers []Layer `json:"layers"`
}

// -----------------------------
// Конструктор сети
// -----------------------------
func NewNetwork(sizes []int) *Network {
	rand.Seed(time.Now().UnixNano())

	layers := make([]Layer, len(sizes)-1)
	for i := 0; i < len(sizes)-1; i++ {
		in := sizes[i]
		out := sizes[i+1]

		w := make([][]float64, out)
		b := make([]float64, out)
		for o := 0; o < out; o++ {
			b[o] = rand.NormFloat64() * 0.1
			w[o] = make([]float64, in)
			for j := 0; j < in; j++ {
				w[o][j] = rand.NormFloat64() * 0.1
			}
		}

		layers[i] = Layer{
			In:      in,
			Out:     out,
			Weights: w,
			Biases:  b,
			Z:       make([]float64, out),
			A:       make([]float64, out),
		}
	}

	return &Network{Layers: layers}
}

// -----------------------------
// Функция активации
// -----------------------------
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	s := sigmoid(x)
	return s * (1 - s)
}

// -----------------------------
// Forward pass
// -----------------------------
func (net *Network) ForwardFull(input []float64) ([]float64, [][]float64, [][]float64) {
	activations := [][]float64{input}
	zvals := [][]float64{}
	a := input

	for i := range net.Layers {
		layer := &net.Layers[i]
		z := make([]float64, layer.Out)
		a2 := make([]float64, layer.Out)

		for j := 0; j < layer.Out; j++ {
			sum := layer.Biases[j]
			for k := 0; k < layer.In; k++ {
				sum += layer.Weights[j][k] * a[k]
			}
			z[j] = sum
			a2[j] = sigmoid(sum)
			layer.Z[j] = sum
			layer.A[j] = a2[j]
		}

		zvals = append(zvals, z)
		activations = append(activations, a2)
		a = a2
	}

	return a, activations, zvals
}

// -----------------------------
// Backpropagation
// -----------------------------
func (net *Network) Backpropagate(target []float64, activations [][]float64, zvals [][]float64) [][]float64 {
	L := len(net.Layers)
	deltas := make([][]float64, L)

	last := L - 1
	outAct := activations[last+1]
	outZ := zvals[last]

	deltas[last] = make([]float64, len(outAct))
	for i := range deltas[last] {
		deltas[last][i] = (outAct[i] - target[i]) * sigmoidPrime(outZ[i])
	}

	for l := L - 2; l >= 0; l-- {
		nextLayer := net.Layers[l+1]
		z := zvals[l]
		deltas[l] = make([]float64, len(z))

		for i := range z {
			sum := 0.0
			for k := range nextLayer.Biases {
				sum += nextLayer.Weights[k][i] * deltas[l+1][k]
			}
			deltas[l][i] = sum * sigmoidPrime(z[i])
		}
	}

	return deltas
}

// -----------------------------
// Apply gradients (коротко)
// -----------------------------
func (net *Network) ApplyGradients(lr float64, deltas [][]float64, activations [][]float64) (maxWeightChange, maxBiasChange float64) {
	for l := range net.Layers {
		layer := &net.Layers[l]
		for i := 0; i < layer.Out; i++ {
			oldBias := layer.Biases[i]
			layer.Biases[i] -= lr * deltas[l][i]
			biasChange := math.Abs(layer.Biases[i] - oldBias)
			if biasChange > maxBiasChange {
				maxBiasChange = biasChange
			}

			for j := 0; j < layer.In; j++ {
				oldWeight := layer.Weights[i][j]
				layer.Weights[i][j] -= lr * deltas[l][i] * activations[l][j]
				weightChange := math.Abs(layer.Weights[i][j] - oldWeight)
				if weightChange > maxWeightChange {
					maxWeightChange = weightChange
				}
			}
		}
	}
	return
}

// -----------------------------
// Predict — простой forward pass
// -----------------------------
func (net *Network) Predict(input []float64) []float64 {
	a := input
	for _, layer := range net.Layers {
		next := make([]float64, layer.Out)
		for i := 0; i < layer.Out; i++ {
			sum := layer.Biases[i]
			for j := 0; j < layer.In; j++ {
				sum += layer.Weights[i][j] * a[j]
			}
			next[i] = sigmoid(sum)
		}
		a = next
	}
	return a
}

// -----------------------------
// TrainBatch с кратким логом по эпохам
// -----------------------------
// -----------------------------
// TrainMiniBatchSGD
// -----------------------------
func (net *Network) TrainMiniBatchSGD(samples [][]float64, targets [][]float64, epochs int, lr float64, batchSize int, f *os.File) {
	n := len(samples)
	rand.Seed(time.Now().UnixNano())

	// --- Параметры ---
	lambda := 0.001       // L2 регуляризация
	patience := 500       // эпох без улучшения для Early Stopping
	minEpochs := 100      // минимальное число эпох перед проверкой Early Stopping

	bestLoss := math.MaxFloat64
	wait := 0

	for e := 0; e < epochs; e++ {
		// Перемешиваем данные каждый раз
		perm := rand.Perm(n)

		totalLoss := 0.0
		var maxWeightChange, maxBiasChange float64

		for start := 0; start < n; start += batchSize {
			end := start + batchSize
			if end > n {
				end = n
			}

			for _, idx := range perm[start:end] {
				x := samples[idx]
				y := targets[idx]

				out, activations, zvals := net.ForwardFull(x)

				// Считаем loss
				sampleLoss := 0.0
				for i := range y {
					diff := out[i] - y[i]
					sampleLoss += 0.5 * diff * diff
				}
				totalLoss += sampleLoss

				// Градиенты
				deltas := net.Backpropagate(y, activations, zvals)

				// Применяем градиенты с L2 регуляризацией
				for l := range net.Layers {
					layer := &net.Layers[l]
					for i := 0; i < layer.Out; i++ {
						oldBias := layer.Biases[i]
						layer.Biases[i] -= lr * deltas[l][i]
						biasChange := math.Abs(layer.Biases[i] - oldBias)
						if biasChange > maxBiasChange {
							maxBiasChange = biasChange
						}

						for j := 0; j < layer.In; j++ {
							oldWeight := layer.Weights[i][j]
							layer.Weights[i][j] -= lr * (deltas[l][i]*activations[l][j] + lambda*layer.Weights[i][j])
							weightChange := math.Abs(layer.Weights[i][j] - oldWeight)
							if weightChange > maxWeightChange {
								maxWeightChange = weightChange
							}
						}
					}
				}
			}
		}

		avgLoss := totalLoss / float64(n)
		// --- Логируем и сбрасываем буфер ---
		fmt.Fprintf(f, "Epoch %d avg loss: %.6f | max weight change: %.6f | max bias change: %.6f\n",
			e, avgLoss, maxWeightChange, maxBiasChange)
		f.Sync() // обязательно сбросить буфер на диск

		// --- Early Stopping ---
		if e >= minEpochs {
			if avgLoss < bestLoss {
				bestLoss = avgLoss
				wait = 0
			} else {
				wait++
				if wait >= patience {
					fmt.Fprintf(f, "Ранняя остановка запущена для эпох %d (нет улучшений для %d эпох)\n", e, patience)
					f.Sync()
					break
				}
			}
		}
	}
}


