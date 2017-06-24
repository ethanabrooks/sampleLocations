package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
)

func randNormVector(size int) *mat64.Vector {
	vector := mat64.NewVector(size, nil)
	for i := 0; i < size; i++ {
		vector.SetVec(i, rand.NormFloat64()*0.001)
	}
	return vector
}

func setRow(i int, m *mat64.Dense, v *mat64.Vector) {
	_, dim := m.Dims()
	for j := 0; j < dim; j++ {
		m.Set(i, j, v.At(j, 0))
	}
}

func randomWalk(steps int, dim int) *mat64.Dense {
	positions := mat64.NewDense(steps, dim, nil)
	newPos := mat64.NewVector(dim, nil)
	vel := mat64.NewVector(dim, nil)
	for i := 1; i < steps; i++ {
		acc := randNormVector(dim)         // noisey acceleration vector
		vel.AddVec(vel, acc)               // vel += acc
		noise := rand.ExpFloat64() / 0.5   // makes path jagged
		vel.ScaleVec(noise, vel)           // vel *= noise
		oldPos := positions.RowView(i - 1) // oldPos = positions[i - 1]
		newPos.AddVec(oldPos, vel)         // newPos = oldPos + vel
		setRow(i, positions, newPos)       // positions[i] = newPos
	}
	return positions
}

func euclidean_distance(a *mat64.Vector, b *mat64.Vector) float64 {
	rowsA, colsA := a.Dims()
	rowsB, colsB := b.Dims()
	if rowsA != rowsB {
		panic(fmt.Sprintf("rows of `a` (%d) not equal to rows of `b` (%d)",
			rowsA, rowsB))
	}
	if colsA != colsB {
		panic(fmt.Sprintf("cols of `a` (%d) not equal to cols of `b` (%d)",
			colsA, colsB))
	}
	sq_distance := 0.0
	for i := 0; i < rowsA; i++ {
		sq_distance += math.Pow(a.At(i, 0)+b.At(i, 0), 2) // (a[i] + b[i])^2
	}
	return math.Sqrt(sq_distance)
}

func get_cost(path *mat64.Dense) float64 {
	size, _ := path.Dims()
	head := path.RowView(0)
	cost := 0.0
	for i := 0; i < size; i++ {
		pos := path.RowView(i)
		cost += euclidean_distance(head, pos)
	}
	return cost
}

//func best_choice(path *mat64.Dense, n_choices int) []int {
//size, _ := path.Dims()
//cache := mat64.NewTriDense(size, false, nil)
//cache.SetTri(0, 0, get_cost(path.
//for i := 1; i < size; i++ {

//}
//}

func main() {
	//walk := randomWalk(3, 1)
	//cost := get_cost(walk)
	//fmt.Println(mat64.Formatted(walk))
	//fmt.Println(cost)
}
