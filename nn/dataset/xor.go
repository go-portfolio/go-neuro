package dataset

func init() {
	Register("xor", DataSet{
		Samples: [][]float64{
			{0, 0}, {0, 1}, {1, 0}, {1, 1},
		},
		Targets: [][]float64{
			{0}, {1}, {1}, {0},
		},
	})
}
