mkdir weights_for_pspnet

pspnet_model.compile(optimizer=optimizer_adam, loss=bce_jaccard_loss) #compilation with all weight on scoremap

#TensorBoard
writer = tf.summary.create_file_writer("./logs_with_Jaccard_pspnet")



for epoch in tqdm(range(25)):


  meteor_train = DataGeneratorPspnet(batch_size=16, dataframe=bbox_frame[:8000], input_size = 480, shuffle=True)
  myGen_train = meteor_train.name_generation(16)
  myGen_train = preprocess_input(myGen_train)
  


  meteor_test = DataGeneratorPspnet(batch_size=16, dataframe=bbox_frame[8000:], input_size = 480, shuffle=True)
  myGen_test = meteor_test.name_generation(16)
  myGen_test = preprocess_input(myGen_test)

  list_of_history_0 = []

  for ij in (range(500)):

    train_batch = next(myGen_train)
    #print('train_batch shape is:',train_batch.shape)
    #break
    #print(train_batch[0].shape)
    #print(train_batch[1].shape)
    history = pspnet_model.train_on_batch(x=train_batch[0], y=train_batch[1])
    list_of_history_0.append(history)

  mean_list_of_history_0 = np.mean(list_of_history_0)

  with writer.as_default():
    tf.summary.scalar("training_loss", mean_list_of_history_0, step=epoch)
    writer.flush()

    
  list_of_result_0 = []

  for ik in (range(125)):

    test_batch = next(myGen_test)
    result = pspnet_model.test_on_batch(x=test_batch[0], y=test_batch[1])
    list_of_result_0.append(result)

  mean_list_of_result_0 = np.mean(list_of_result_0)

  with writer.as_default():
    tf.summary.scalar("validation_loss", mean_list_of_result_0, step=epoch)
    writer.flush()


  # Save weights to HDF5 at each epoch

  pspnet_model.save_weights(f"weights_for_pspnet/pspnet_model_with_Jaccard_epoch{epoch}.h5")
  print(f"pspnet_model_with_Jaccard_epoch{epoch} saved to weights_for_pspnet")

