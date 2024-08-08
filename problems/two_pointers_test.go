package problems

import "testing"

func TestSortColors(t *testing.T) {
	type args struct {
		nums []int
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "Test 1",
			args: args{
				nums: []int{2, 0, 2, 1, 1, 0},
			},
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			SortColors(tt.args.nums)
		})
	}
}
