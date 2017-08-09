require 'matrix'
require "byebug"

# Matrix element wise matrix multiplication
# Hadamard product (matrices):
# https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
def element_multiplication(m1, m2)
	m3 = Matrix.build(m1.row_count, m1.column_count) {|r, c| m1[r, c] * m2[r, c]}
	return m3
end

# Summation of all values in a matrix
def element_summation(m)
	s = 0
	m.each {|x| s += x}
	return s
end

# a confusion matrix illustrates the accuracy of a classifier:
# values in the diagonal of the classifier are correctly classified.
# https://en.wikipedia.org/wiki/Confusion_matrix
def confusion_matrix(expected, predicted)
	expected = expected.to_a.map {|x| x.index(x.max)}
	predicted = predicted.to_a.map {|x| x.index(x.max)}

	n = (expected + predicted).uniq.length
	cm = Matrix.build(n){0}.to_a
	# byebug
	expected.zip(predicted).map {|x, y| cm[x][y]+=1}

	return Matrix.rows(cm)
end

#The actual neural network
class NeuralNetwork
	def initialize()
		# For the sake of simplicity feel free to hardcode
		# parameters. The goal is a working feedforward neral
		# network. One hidden layer, one input layer, and one
		# output layer are enough to achieve 99% accuracy on
		# the data set.
		@eta = 0.1
		@hidden_nodes = 3
		@iteration = 500
		# initial weights are in the range of [-init_w_range, +init_w_range]
		@init_w_range = 0.1
	end

	##############################################
	def train(x, y)
		# the training method that updates the internal weights
		# using the predict
		@weights_ij, @weights_ij = back_and_forward_prop(x, y)
	end

	##############################################
	def predict(x)
	end

	##############################################
	protected
		def back_and_forward_prop(x, y)
			# byebug
			x0 = Array.new(x.row_size, 1)
			x = Matrix.columns(x.transpose.to_a.unshift(x0))
			weights_ij = init_weights(x.column_size, @hidden_nodes)
			weights_jk = init_weights(@hidden_nodes+1, 1)
			@iteration.times do
				row_to_pick = rand(x.row_size+1)
				x_rand = Matrix.rows([x.row(row_to_pick)])
				y_rand = y.row(row_to_pick)
				x_hidden, output, s_output, s_hidden = propagate(x_rand, weights_ij, weights_jk)
				delta_hidden, delta_output = back_propagate(x_rand, y_rand, x_hidden, output, weight_jk, S_output, S_hidden)
				grad_matrix_ij = @eta * (delta_hidden * x_rand).transpose
				weights_ij -= grad_matrix_ij
				grad_matrix_jk = @eta * (delta_output * x_hidden)
				weights_jk -= grad_matrix_jk.transpose
			end
			[weights_ij, weights_jk]
		end

		##############################################
		def propagate(x, weights_ij, weights_jk)
			# applies the input to the network
			# this is the forward propagation step
			s_hidden = x * weights_ij
			x_hidden = Matrix.rows([[1] + s_hidden.row(0).map { |e| Math.tanh(e) }.to_a])
			s_output = x_hidden * weights_jk
			byebug
			output = Math.tanh(s_output.row(0)[0])
		end

		##############################################
		def back_propagate(x, y, y_hat)
			# goes backwards and finds the weights
			# that need to be tuned
		end

	private
		def init_weights(rows, cols)
			rand_num = 2 * @init_w_range * rand - @init_w_range
			Matrix.rows(Array.new(rows) { Array.new(cols) { rand_num } })
		end
end
