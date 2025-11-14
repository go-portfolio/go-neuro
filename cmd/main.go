package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/go-portfolio/go-neuro/nn"
	"github.com/go-portfolio/go-neuro/nn/dataset"
	"github.com/go-portfolio/go-neuro/nn/trainers"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Выбираем датасет
	ds := dataset.MustGet("and")

	// Архитектура нейросети
	net := nn.NewNetwork([]int{2, 2, 1})

	// Выбираем алгоритм обучения
	trainer := &trainers.SGDTrainer{LearningRate: 0.5}

	// Обучение
	net.Fit(trainer, ds.Samples, ds.Targets, 20000)

	// Тест
	fmt.Println("\nРезультаты:")
	for i, s := range ds.Samples {
		out := net.Predict(s)
		fmt.Printf("Вход: %v → %.4f (округлён %d), Цель: %v\n",
			s, out[0], int(math.Round(out[0])), ds.Targets[i])
	}
}
