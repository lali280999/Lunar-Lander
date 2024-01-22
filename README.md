Install swig:
    sudo apt-get install swig

Install dependencies:
    pip install -r requirements.txt

To train run:
    python3 lunar_rl_deepq.py

To test run (note that it requires model_new7.pth file to load the dataset):
    python3 lunar_rl_deepq_test.py

To plot the rewards run (note that it requires mean_total_rewards_new7.npy and list_of_total_rewards_new7npy.npy to run)
    python3 plot.py