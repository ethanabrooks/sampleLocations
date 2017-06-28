package main

import (
	"github.com/gonum/matrix/mat64"
	"math/rand"
	"log"
)

// RandNormVector generates a random vector size `int`.
func randNormVector(size int) *mat64.Vector {
	vector := mat64.NewVector(size, nil)
	for i := 0; i < size; i++ {
		vector.SetVec(i, rand.NormFloat64()*0.001)
	}
	return vector
}

// SetRow sets row `i` of `m` to the values in `v`.
func setRow(i int, m *mat64.Dense, v *mat64.Vector) {
	dimV, _ := v.Dims()
	sizeM, dimM := m.Dims()
	if dimV != dimM {
		log.Fatalf("`v` is size %d. " +
			"It must be size %d because `m` is %d x %d.",
			dimV, dimM, sizeM, dimM)
	}
	for j := 0; j < dimM; j++ {
		m.Set(i, j, v.At(j, 0))
	}
}

// SimpleRandomWalk generates a random path length `steps`
// containing all whole number values.
func simpleRandomWalk(steps int) *mat64.Dense {
	positions := mat64.NewDense(steps, 1, nil)
	vel := 0
	for i := 1; i < steps; i++ {
		acc := rand.Intn(5)
		if rand.NormFloat64() > 0 {
			vel += acc
		} else {
			vel -= acc
		}
		oldPos := int(positions.At(i-1, 0))
		positions.Set(i, 0, float64(oldPos+vel))
	}
	return positions
}

// RandomWalk generates a random path with dimensions `steps` x `dim`.
func randomWalk(steps int, dim int) *mat64.Dense {
	positions := mat64.NewDense(steps, dim, nil)
	newPos := mat64.NewVector(dim, nil)
	vel := mat64.NewVector(dim, nil)
	for i := 1; i < steps; i++ {
		acc := randNormVector(dim)         // noisy acceleration vector
		vel.AddVec(vel, acc)               // vel += acc
		noise := rand.ExpFloat64() / 0.5   // makes path jagged
		vel.ScaleVec(noise, vel)           // vel *= noise
		oldPos := positions.RowView(i - 1) // oldPos = positions[i - 1]
		newPos.AddVec(oldPos, vel)         // newPos = oldPos + vel
		setRow(i, positions, newPos)       // positions[i] = newPos
	}
	return positions
}
