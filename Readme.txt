There are 2 models: Generative and Retrieval

Generative Model:
	1 LSTM: it is lstm model trained on movie dialogue corpus. 
		Training - Chatbot_generative_train_LSTM.py
		Testing - Chatbot_generative_test_LSTM.py

	2 LSTM: it is lstm model trained on ubuntu dialogue corpus. 
		Data Processing - Generative_Model_ubuntuDataProcessing.py 				Training - Chatbot_generative_train_Ubuntu.py 
		Testing - Chatbot_generative_test_Ubuntu.py

	3 GRU: it is gru model trained on movie dialogue corpus. 
		Training - Chatbot_generative_train_GRU.py
		Testing - Chatbot_generative_test_GRU.py

Retrieval Model:
	Data Processing - Retrieval_Model_ubuntuDataProcessing.py 				Training - Chatbot_retrieval_train.py
	Testing - Chatbot_retrieval_test.py
