@uthor: me_teor21
d@te: 21/12/2020 10:55 pm



def weighted_crossentropy(truemapbidon, scoremapbidon):


  ''' this function calculates the weighted crossentropy for maps of shape=(2, 3, 3) '''

  sommede_scoremapbidon = tf.reduce_sum(scoremapbidon, axis=[1, 2]).numpy()  #sum over axis1  axis2
  sommede_truemapbidon = tf.reduce_sum(truemapbidon, axis=[1, 2]).numpy()  #sum over axis1  axis2

  CARD_true_map1 = truemapbidon.shape[1] * truemapbidon.shape[2]   #compute the area of map1 (number of elements)
  CARD_true_map2 = truemapbidon.shape[1] * truemapbidon.shape[2]   #compute the area of map2 (number of elements)

  sigma_map1 = sommede_truemapbidon[0]  #sum of the elements
  sigma_map2 = sommede_truemapbidon[1]  #sum of the elements

  betabidon1 = 1 - (sigma_map1/CARD_true_map1)   #compute the beta factor
  betabidon2 = 1 - (sigma_map2/CARD_true_map2)   #compute the beta factor

  #Y* * log(Y_hat)
  ystar_minus_log_yhat = truemapbidon * tf.math.log(scoremapbidon)
  red_sum_of_ystar_minus_log_yhat = tf.reduce_sum(ystar_minus_log_yhat, axis=[1, 2]).numpy()
  aver_red_sum_of_ystar_minus_log_yhat = sum(red_sum_of_ystar_minus_log_yhat)/2

  #(1- Y*) * (1 - log(Y_hat))
  oneminusoneminusystar_minus_log_yhat = (1 - truemapbidon) * (1 - tf.math.log(scoremapbidon))
  red_sum_of_oneminusoneminusystar_minus_log_yhat = tf.reduce_sum(oneminusoneminusystar_minus_log_yhat, axis=[1, 2]).numpy()
  aver_red_sum_of_oneminusoneminusystar_minus_log_yhat = sum(red_sum_of_oneminusoneminusystar_minus_log_yhat)/2

  mean_of_beta = (betabidon1 + betabidon2)/2   #average over the batches

  #apply the final step of the calculation of the formula
  loss_scoremap = -mean_of_beta * aver_red_sum_of_ystar_minus_log_yhat - (1-mean_of_beta) * aver_red_sum_of_oneminusoneminusystar_minus_log_yhat 


  return loss_scoremap





def quad_norm(g_true, g_pred):
  ''' this function calculates the distance for maps of shape=(n, 3, 3, 4) '''

    nb_batch = 2   #arrange according your case 
    diff = g_true - g_pred    #difference between true and pred
    square = tf.square(diff)  #compute the sqaure
    sumdiffsquare = tf.reduce_sum(square, axis=[1,2]).numpy()  #sum over the faces
    sumdiffsquareaxis1 = ((tf.reduce_sum(sumdiffsquare, axis=1).numpy())/4)    #sum over axis1 and average over 4
    averagesumdiffaxis1 = (sum(sumdiffsquareaxis1)/nb_batch)    #average over the batches
    distance = tf.sqrt(averagesumdiffaxis1).numpy()   #take the squareroot  real distance
    loss_QUADgeo = distance

    return distance





def loss(loss_scoremap, loss_QUADgeo):
  '''This function will compute our tatl loss based on the scoremap and QUAD losses'''

  theLambda = 1  #arrange according the importance of both losses
  total_loss = loss_scoremap + theLambda*loss_QUADgeo #linear combination of both losses
  
  return total_loss





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
