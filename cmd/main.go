package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/go-portfolio/go-neuro/nn"
)

func main() {
	rand.Seed(time.Now().UnixNano())

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

	net := nn.NewNetwork([]int{2, 2, 1})
	epochs := 20000
	learningRate := 0.5

	net.Train(samples, targets, epochs, learningRate)

	fmt.Println("\nРезультаты после обучения:")
	for i, s := range samples {
		out := net.Predict(s)
		fmt.Printf("Вход: %v → Выход: %.4f (округлён %d), Цель: %v\n",
			s, out[0], int(math.Round(out[0])), targets[i])
	}
}
