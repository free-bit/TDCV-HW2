To generate executable:
- Make sure release/ exists
- Go to release/
- cmake -DCMAKE_BUILD_TYPE=Release ..
- make

Run executable for training + testing:
- Make sure ../data/ exists
- Make sure task3/preds/ exists
- Go to task3/release/
- ./task3 [--train]

Run executable for testing only:
- Make sure task3/preds/ exists
- Make sure task3/model/ exists and files related to model is there 
- Go to task3/release/
- ./task3 --load

Plot PR:
- Go to task3/
- Make sure task3/pc_data.txt exists
- python3 plot.py

*Best results achieved is put under results/ folder along with the files holding the state of trained model.