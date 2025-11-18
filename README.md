# TTLS

The code is implemented based on RecBole, which must be downloaded first: https://github.com/RUCAIBox/RecBole               

#### The following are the modifications made to RecBole/recbole:：          
1.backbone: add modifications for the original model loss.               
2.recbole_data: add data splitting section.            
3.recbole_train: trainer.py add evaluate_ttt partition, The other two files control the training process.                     
The above can directly replace the original files of RecBole/recbole.     
#### The following are new files：           
4.merge_model: Model Merging, initiated by quick_run.           
5.run_fine_tuning: Fine-tuning, initiated by quick_run.      
  
       
 
