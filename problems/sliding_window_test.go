package problems

import "testing"

func TestMinWindow(t *testing.T) {
	type args struct {
		s string
		t string
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{name: "Testcase 1", args: args{s: "ADOBECODEBANC", t: "ABC"}, want: "BANC"},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MinWindow(tt.args.s, tt.args.t); got != tt.want {
				t.Errorf("MinWindow() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCheckInclusion(t *testing.T) {
	type args struct {
		s1 string
		s2 string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{name: "Testcase 1", args: args{s1: "ab", s2: "eidboaooo"}, want: false},
		{name: "Testcase 2", args: args{s1: "ab", s2: "eidbaooo"}, want: true},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CheckInclusion(tt.args.s1, tt.args.s2); got != tt.want {
				t.Errorf("CheckInclusion() = %v, want %v", got, tt.want)
			}
		})
	}
}
