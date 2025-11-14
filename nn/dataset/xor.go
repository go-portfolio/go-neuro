package dataset

func init() {
    Register(Dataset{
        Name:     "xor",
        TaskType: "logic",
        Samples: [][]float64{
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1},
        },
        Targets: [][]float64{
            {0}, {1}, {1}, {0},
        },
    })
}
