package dataset

func init() {
	Register("and", DataSet{
		Samples: [][]float64{
			{0, 0}, {0, 1}, {1, 0}, {1, 1},
		},
		Targets: [][]float64{
			{0}, {0}, {0}, {1},
		},
	})
}
