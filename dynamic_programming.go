package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math/rand"
)

func randomWalk(steps int, dim int) *mat64.Dense {
	positions := mat64.NewDense(steps, dim, nil)
	newPos := mat64.NewVector(dim, nil)
	vel := mat64.NewVector(dim, nil)
	for i := 1; i < steps; i++ {

		// noisey acceleration vector
		acc := mat64.NewVector(dim, nil)
		for j := 0; j < dim; j++ {
			acc.SetVec(j, rand.NormFloat64()*0.001)
		}

		// vel += acc
		vel.AddVec(vel, acc)

		// vel *= noise
		vel.ScaleVec(rand.ExpFloat64()/0.5, vel)

		// oldPos = positions[i - 1]
		oldPos := positions.RowView(i - 1)

		// newPos = oldPos + vel
		newPos.AddVec(oldPos, vel)

		// positions[i] = newPos
		for j := 0; j < dim; j++ {
			positions.Set(i, j, newPos.At(j, 0))
		}
	}
	return positions
}

func main() {
	walk := randomWalk(5, 2)
	for i := 0; i < 5; i++ {
		fmt.Println(walk.RawRowView(i))
	}
}
