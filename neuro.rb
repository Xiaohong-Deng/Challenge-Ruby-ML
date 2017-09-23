require 'matrix'
require "byebug"

class Matrix
	public :[]=
end
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
	expected.zip(predicted).map {|x, y| cm[x][y]+=1}

	return Matrix.rows(cm)
end

def accuracy(cm)
	cm.trace.to_f / element_summation(cm)
end

#The actual neural network
class NeuralNetwork
	def initialize()
		# For the sake of simplicity feel free to hardcode
		# parameters. The goal is a working feedforward neral
		# network. One hidden layer, one input layer, and one
		# output layer are enough to achieve 99% accuracy on
		# the data set.
		@eta = 0.01
		@hidden_nodes = 6
		@iteration = 50000
		# initial weights are in the range of [-init_w_range, +init_w_range]
		@init_w_range = 0.1
	end

	##############################################
	def train(x, y)
		# the training method that updates the internal weights
		# using the predict
		@weights_ij, @weights_jk = back_and_forward_prop(x, y)
	end

	##############################################
	def predict(x)
		x0 = Array.new(x.row_size, 1)
		x = Matrix.columns(x.transpose.to_a.unshift(x0))
		s_hidden = x * @weights_ij
		x_hidden = Matrix.rows(Array.new(s_hidden.row_size, Array.new(s_hidden.column_size, 0)))
		s_hidden.each_with_index do |e, row, col|
			x_hidden[row, col] = Math.tanh(e)
		end
		x0_hidden = Array.new(x_hidden.row_size, 1)
		x_hidden = Matrix.columns(x_hidden.transpose.to_a.unshift(x0_hidden))
		s_output = x_hidden * @weights_jk
		y_predict = Matrix.rows(Array.new(s_output.row_size, Array.new(s_output.column_size, 0)))
		s_output.each_with_index do |e, row, col|
			y_predict[row, col] = Math.tanh(e)
		end

		# set predicted values to 1's and 0's
		# y_columns_num = y_predict.column_size
		# y_predict.row_size.times do |row_num|
		# 	row, col = y_predict.index(y_predict.row(row_num).max)
		# 	y_predict[row, col] = 1
		# 	y_columns_num.times do |current_col_num|
		# 		if current_col_num != col
		# 			y_predict[row, current_col_num] = 0
		# 		end
		# 	end
		# end
		# byebug
		y_predict
	end

	##############################################
	protected
		def back_and_forward_prop(x, y)
			x0 = Array.new(x.row_size, 1)
			x = Matrix.columns(x.transpose.to_a.unshift(x0))
			weights_ij = init_weights(x.column_size, @hidden_nodes)
			weights_jk = init_weights(@hidden_nodes+1, 3)
			@iteration.times do
				row_to_pick = rand(x.row_size)
				x_rand = Matrix.rows([x.row(row_to_pick)])
				y_rand = Matrix.rows([y.row(row_to_pick)])
				x_hidden, output, s_output, s_hidden = propagate(x_rand, weights_ij, weights_jk)
				delta_hidden, delta_output = back_propagate(x_rand, y_rand, x_hidden, output, weights_jk, s_output, s_hidden)
				grad_matrix_ij = @eta * (x_rand.transpose * delta_hidden)
				weights_ij -= grad_matrix_ij
				grad_matrix_jk = @eta * (x_hidden.transpose * delta_output)
				weights_jk -= grad_matrix_jk
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
			output = Matrix.rows([s_output.row(0).map { |e| Math.tanh(e) }])
			[x_hidden, output, s_output, s_hidden]
		end

		##############################################
		def back_propagate(x, y, x_hidden, output, weights_jk, s_output, s_hidden)
			# goes backwards and finds the weights
			# that need to be tuned
			delta_output = Matrix.rows(Array.new(y.row_size, Array.new(y.column_size, 0)))
			derv_tanh_s_op = derv_tanh(s_output)
			output.each_with_index { |e, row, col| delta_output[row, col] = -2 * (y[row, col] - e) * derv_tanh_s_op[row, col] }
			# no delta for constant node because j is the number of hidden nodes in wij, constant node is constant, no need for w
			# to compute it, so we only need #hidden nodes of delta for hidden layer
			dh_size = x_hidden.column_size - 1
			delta_hidden = Matrix.rows(Array.new(1, Array.new(dh_size, 0)))
			derv_tanh_s_hd = derv_tanh(s_hidden)
			dh_size.times do |hd_node_index|
				delta_output.each_with_index do |e, row, col|
					delta_hidden[row, hd_node_index] += e * weights_jk[hd_node_index, col] * derv_tanh_s_hd[row, hd_node_index]
				end
			end
			[delta_hidden, delta_output]
		end

	private
		def init_weights(rows, cols)
			rand_num = lambda { 2 * @init_w_range * rand - @init_w_range }
			Matrix.rows(Array.new(rows) { Array.new(cols) { rand_num.call } })
		end

		def derv_tanh(s)
			tanh_s = s.map { |e| Math.tanh(e) }
			tanh_s.map { |e| 1 - e * e }
		end
end
