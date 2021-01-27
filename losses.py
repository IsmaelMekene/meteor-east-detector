@uthor: me_teor21
d@te: 21/12/2020 10:55 pm



def weighted_crossentropy(truemap, scoremap, eps = 1e-5):


  ''' this function calculates the weighted crossentropy for maps of shape=(n, 3, 3) '''
  #eps = 1e-1

  sommede_scoremap = tf.reduce_sum(scoremap, axis=[1, 2])  #sum over axis1  axis2
  sommede_truemap = tf.reduce_sum(truemap, axis=[1, 2])  #sum over axis1  axis2

  CARD_true_map = truemap.shape[1] * truemap.shape[2]   #compute the area of maps (number of elements)
  
  thebeta = 1-(sommede_truemap/CARD_true_map)


  #Y* * log(Y_hat)
  ystar_minus_log_yhat = truemap * tf.math.log(scoremap+eps)
  red_sum_of_ystar_minus_log_yhat = tf.reduce_sum(ystar_minus_log_yhat, axis=[1, 2])
  aver_red_sum_of_ystar_minus_log_yhat = tf.reduce_mean(red_sum_of_ystar_minus_log_yhat)

  #(1- Y*) * (1 - log(Y_hat))
  oneminusoneminusystar_minus_log_yhat = (1 - truemap) * (1 - tf.math.log(scoremap+eps))
  red_sum_of_oneminusoneminusystar_minus_log_yhat = tf.reduce_sum(oneminusoneminusystar_minus_log_yhat, axis=[1, 2])
  aver_red_sum_of_oneminusoneminusystar_minus_log_yhat = tf.reduce_mean(red_sum_of_oneminusoneminusystar_minus_log_yhat)

  #average over the batches
  mean_of_beta = tf.reduce_mean(thebeta)

  #apply the final step of the calculation of the formula
  loss_scoremap = -mean_of_beta * aver_red_sum_of_ystar_minus_log_yhat - (1-mean_of_beta) * aver_red_sum_of_oneminusoneminusystar_minus_log_yhat 

  #print(loss_scoremap)
  #print(thebeta)
  return loss_scoremap





def quad_norm(g_true, g_pred):
  ''' this function calculates the distance for maps of shape=(n, m, m, 4) '''

  nb_batch = g_true.shape[0]   #arrange according your case 
  diff = g_true - g_pred    #difference between true and pred
  square = tf.square(diff)  #compute the sqaure
  sumdiffsquare = tf.reduce_sum(square, axis=[1,2])  #sum over the faces
  sumdiffsquareaxis1 = tf.reduce_sum(sumdiffsquare, axis=1)/4    #sum over axis1 and average over 4
  averagesumdiffaxis1 = tf.reduce_mean(sumdiffsquareaxis1)    #average over the batches
  distance = tf.sqrt(averagesumdiffaxis1)   #take the squareroot  real distance
  loss_QUADgeo = distance

  return distance




def smooth_l1_loss(prediction_tensor, target_tensor, weights):
  '''This function compute the smooth_l1_loss'''
  
  n_q = tf.reshape(quad_norm(target_tensor), tf.shape(weights))
  diff = prediction_tensor - target_tensor
  abs_diff = tf.abs(diff)
  abs_diff_lt_1 = tf.less(abs_diff, 1)
  pixel_wise_smooth_l1norm = (tf.reduce_sum(
      tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),
      axis=-1) / n_q) * weights
  return pixel_wise_smooth_l1norm



#losses

losses = {
	"outputlayer_SCORE": weighted_crossentropy,
	"QUAD_geometry": quad_norm
}
lossWeights = {"outputlayer_SCORE": 1.0, "QUAD_geometry": 1.0}
      

    
