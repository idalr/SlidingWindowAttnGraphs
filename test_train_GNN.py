# a COPY of `train_GNN.py`
# BUT on a smaller scale, # mini version of 50 input
# running without a YAML file on a local machine # explicitly stated configuration
# excluding  heuristics setting # will be commented out


import yaml
import argparse
from graph_data_loaders import AttentionGraphs, HeuristicGraphs
import torch
import os
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from nltk.tokenize import sent_tokenize
from gnn_model import GAT_model, GCN_model, partitions
from base_model import MHAClassifier
from eval_models import retrieve_parameters, eval_results
from preprocess_data import load_data
from data_loaders import create_loaders, get_class_weights

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import F1Score


os.environ["TOKENIZERS_PARALLELISM"] = "False"

def main_run(config_file , settings_file):
    os.environ["CUDA_VISIBLE_DEVICES"]= "0" #config_file["cuda_visible_devices"]
    logger_name = "df_logger_cw.csv" # config_file["logger_name"] #"df_logger_cw.csv"
    model_name = "Extended_NoTemp_100" # config_file["model_name"]  #Extended_Anneal
    type_model = "GAT" # config_file["type_model"] #GAT or GCN

    root_graph = "AttnGraphs_HND/" # config_file["data_paths"]["root_graph_dataset"] #"/scratch/datasets/AttnGraphs_HND/"
    path_results = "AttGraphs/GNN_Results/" # config_file["data_paths"]["results_folder"] #"/home/mbugueno/AttGraphs/GNN_Results/" #folder for GNN results (Heuristic_Results for baselines)
    path_logger = "HomoGraphs_HND/" # config_file["data_paths"]["path_logger"]  #path_logger = "/scratch/mbugueno/HomoGraphs_HND/"

    if not os.path.exists(root_graph):
        os.makedirs(root_graph)  # Creates the folder
        print(f"Folder '{root_graph}' created.")

    if not os.path.exists(path_results):
        os.makedirs(path_results)  # Creates the folder
        print(f"Folder '{path_results}' created.")

    if not os.path.exists(path_logger):
        os.makedirs(path_logger)  # Creates the folder
        print(f"Folder '{path_logger}' created.")

    num_classes = 2 # config_file["model_arch_args"]["num_classes"]
    lr= 0.001 # config_file["model_arch_args"]["lr"]
    dropout= 0.2# config_file["model_arch_args"]["dropout"]#0.2
    dim_features = [64] # config_file["model_arch_args"]["dim_features"]#[128, 256]
    n_layers = [2] # config_file["model_arch_args"]["n_layers"] #[1, 2]
    num_runs = 1 # config_file["model_arch_args"]["num_runs"] #5
    
    df_logger = pd.read_csv(path_logger+logger_name)    

    # if model_name=="Extended_Sigmoid":
    #     path_checkpoint, model_score = retrieve_parameters(model_name, df_logger, require_best=False, retrieve_index=18)
    # else:
    path_checkpoint, model_score = retrieve_parameters(model_name, df_logger)
    
    
    if config_file["baseline"]:
        pass
    #     heuristic = config_file["heuristic"] #max_semantic, mean_semantic, order, window
    #     path_root = root_graph+"Heuristic/"+heuristic
    #     project_name = type_model+"_"+heuristic
    #     file_results = path_results+project_name
    #     path_models = path_logger+"Heuristic/"
    else:
        file_to_save = model_name+"_"+str(model_score)[:5]
        type_graph = "full" #config_file["type_graph"] #full, mean, max
        path_models = path_logger+model_name+"/"
        
        if config_file["normalized"]:
            pass
            # path_root=root_graph+model_name+"/Attention/"+type_graph+"_norm" ## _norm if normalized
            # project_name= model_name+"2"+type_model+"_"+type_graph+"_norm"
            # file_results = path_results+file_to_save+"_2"+type_model+"_"+type_graph+"_norm"
        else: 
            path_root= os.path.join(root_graph, model_name, "Attention", type_graph) #root_graph+model_name+"/Attention/"+type_graph
            project_name= model_name+"2"+type_model+"_"+type_graph
            file_results = path_results+file_to_save+"_2"+type_model+"_"+type_graph
            
    filename="post_predict_train_documents.csv"
    filename_test= "post_predict_test_documents.csv"

    # TODO: problem with AttentionGraph
    ## calling already processed files shouldn't need path/to/vocab, so none has to be passed, just have to call data in /processed/data.pt
    ## if no processed files, then call the vocab and process data from /raw/data.csv
    ## have to dig into torch_geometric.Dataset to properly define class property

    try:
        if config_file["baseline"]:
            pass
            # dataset = HeuristicGraphs(root=path_root, filename=filename, heuristic=heuristic, path_invert_vocab_sent='')
            # dataset_test = HeuristicGraphs(root=path_root, filename=filename_test, heuristic=heuristic, path_invert_vocab_sent='', test=True)
        else:
            dataset = AttentionGraphs(root=path_root, filename=filename, filter_type="", input_matrices=None, path_invert_vocab_sent='', degree=0.5, test=False)
            dataset_test = AttentionGraphs(root=path_root, filename=filename_test, filter_type="", input_matrices=None, path_invert_vocab_sent='', degree=0.5, test=True)

            # Cause error, in order to test if Exception also runs
            error = AttentionGraphs(root=path_root, filename="error.csv", filter_type='', input_matrices=None, path_invert_vocab_sent='')

        df_full_train, _ = load_data(**config_file["load_data_paths"])
        if config_file["with_cw"]:
            my_class_weights, labels_counter = get_class_weights(df_full_train)
            calculated_cw = my_class_weights
            print ("\nClass weights - from training partition:", my_class_weights)
            print ("Class counter:", labels_counter)
        else:
            print ("\n-- No class weights specificied --\n")
            calculated_cw = None

    except:
        if config_file: # temperary, so no try-except at the moment!
            print ("Error loading dataset - No Graph Dataset found")
            print ("\nCreating new dataset from pre-trained MHA model")
            print ("Pre-trained model:", model_name)
            if config_file["baseline"]:
                pass
            # print ("Heuristic:", heuristic)
            else:
                print ("Type graph:", type_graph)

            print ("Root graph:", path_root)
        
            print ("Loading data...")
            df_full_train, df_test = load_data(**config_file["load_data_paths"])

            sent_lengths=[]
            for i, doc in enumerate(df_full_train['article_text']):
                sent_in_doc = sent_tokenize(doc)
                if len(sent_in_doc)==0:
                    print ("Empty doc en:", i)
                sent_lengths.append(len(sent_in_doc))
            max_len = max(sent_lengths)  # Maximum number of sentences in a document

            ############################################################################# mini run
            df_full_train = df_full_train.head(50)
            df_test = df_test.head(50)
            sent_lengths = sent_lengths[:50]
            ############################################################################# mini run

            loader_train, loader_test, _, _ = create_loaders(df_full_train, df_test, max_len, config_file["batch_size"], with_val=False, task="classification",
                                                                                               tokenizer_from_scratch=False, path_ckpt=config_file["load_data_paths"]["in_path"])
        
            if config_file["with_cw"]:
                my_class_weights, labels_counter = get_class_weights(df_full_train)
                calculated_cw = my_class_weights
                print ("\nClass weights - from training partition:", my_class_weights)
                print ("Class counter:", labels_counter)
            else:
                print ("\n-- No class weights specificied --\n")
                calculated_cw = None

            if config_file["baseline"]:
                model_lightning = MHAClassifier.load_from_checkpoint(path_checkpoint)
            else:
                print ("\nLoading", model_name, "({0:.3f}".format(model_score),") from:", path_checkpoint)
                model_lightning = MHAClassifier.load_from_checkpoint(path_checkpoint)
                print ("Done")

            preds_t, full_attn_weights_t, all_labels_t, all_doc_ids_t, all_article_identifiers_t = model_lightning.predict(loader_train, cpu_store=False)
            preds_test, full_attn_weights_test, all_labels_test, all_doc_ids_test, all_article_identifiers_test = model_lightning.predict(loader_test, cpu_store=False)

            print("Done with predicting")
        
            if not config_file["baseline"]:
                acc_t, f1_all_t = eval_results(preds_t, all_labels_t, num_classes, "Train")
                acc_test, f1_all_test = eval_results(preds_test, all_labels_test, num_classes, "Test")

            path_dataset = path_root+"/raw/"

            print ("\nCreating files for PyG dataset in:", path_dataset)

            post_predict_train_docs = pd.DataFrame(columns=["article_id", "label", "doc_as_ids"])
            post_predict_train_docs.to_csv(path_dataset+filename, index=False)
            for article_id, label, doc_as_ids in zip(all_article_identifiers_t, all_labels_t, all_doc_ids_t):
                post_predict_train_docs.loc[len(post_predict_train_docs)] = {
                "article_id": article_id.item(),
                "label": label.item(),
                "doc_as_ids": doc_as_ids.tolist()
                }
            post_predict_train_docs.to_csv(path_dataset+filename, index=False)
            print ("Finished and saved in:", path_dataset+filename)

            post_predict_test_docs = pd.DataFrame(columns=["article_id", "label", "doc_as_ids"])
            post_predict_test_docs.to_csv(path_dataset+filename_test, index=False)
            for article_id, label, doc_as_ids in zip(all_article_identifiers_test, all_labels_test, all_doc_ids_test):
                post_predict_test_docs.loc[len(post_predict_test_docs)] = {
                "article_id": article_id.item(),
                "label": label.item(),
                "doc_as_ids": doc_as_ids.tolist()
                }
            post_predict_test_docs.to_csv(path_dataset+filename_test, index=False)
            print ("Finished and saved in:", path_dataset+filename_test)

            #### create data.pt
            if config_file["baseline"]:
                pass
                # start_creation = time.time()
                # dataset = HeuristicGraphs(root=path_root, filename=filename, heuristic=heuristic,
                #                           path_invert_vocab_sent=config_file["load_data_paths"]["in_path"], test=False)
                # creation_train = time.time()-start_creation
                # start_creation = time.time()
                # dataset_test = HeuristicGraphs(root=path_root, filename=filename_test, heuristic=heuristic,
                #                                path_invert_vocab_sent=config_file["load_data_paths"]["in_path"], test=True)
                # creation_test = time.time()-start_creation

            else:
                if type_graph=="full":
                    filter_type=None
                else:
                    pass
                    # filter_type=type_graph

                start_creation = time.time()
                dataset = AttentionGraphs(root=path_root, filename=filename, filter_type=filter_type, input_matrices=full_attn_weights_t,
                                      path_invert_vocab_sent=config_file["load_data_paths"]["in_path"], degree=0.5, normalized=config_file["normalized"], test=False) 
                creation_train = time.time()-start_creation
                start_creation = time.time()
                dataset_test = AttentionGraphs(root=path_root, filename=filename_test, filter_type=filter_type, input_matrices=full_attn_weights_test,
                                           path_invert_vocab_sent=config_file["load_data_paths"]["in_path"], degree=0.5, normalized=config_file["normalized"], test=True)
                creation_test = time.time()-start_creation

            #### save creation time + base model results in file_results
            if config_file["baseline"]:
                pass
                # with open(file_results+'.txt', 'a') as f:
                #     print ("================================================", file=f)
                #     print ("Graph Creation Time:", model_name, file=f)
                #     print ("================================================", file=f)
                #     print ("[TRAIN] Dataset creation time: ", creation_train, file=f)
                #     print ("[TEST] Dataset creation time: ", creation_test, file=f)
                #     f.close()

            else:
                with open(file_results+'.txt', 'a') as f:
                    print ("================================================", file=f)
                    print ("Evaluation of pre-trained model:", model_name, file=f)
                    print ("================================================", file=f)
                    print ("[TRAIN] Acc:", acc_t.item(), file=f)
                    print ("[TRAIN] F1-macro:", f1_all_t.mean().item(), file=f)
                    print ("[TRAIN] F1-scores:", f1_all_t, file=f)
                    print ("------------------------", file=f)
                    print ("[TEST] Acc:", acc_test.item(), file=f)
                    print ("[TEST] F1-macro:", f1_all_test.mean().item(), file=f)
                    print ("[TEST] F1-scores:", f1_all_test, file=f)
                    print ("================================================", file=f)
                    print ("[TRAIN] Dataset creation time: ", creation_train, file=f)
                    print ("[TEST] Dataset creation time: ", creation_test, file=f)
                    f.close()

    ### Run GNN models
    start = time.time()
    np.set_printoptions(precision=3)

    if config_file["baseline"]:
        pass
        # graph_construction = heuristic
    else:
        graph_construction = type_graph
    
    with open(file_results+'.txt', 'a') as f:
        """
        ### duplicated
        try:
            print ("[TRAIN] Acc:", acc_t.item(), file=f)
            print ("[TRAIN] F1-macro:", f1_all_t.mean().item(), file=f)
            print ("[TRAIN] F1-scores:", f1_all_t, file=f) #print train pre-trained model
            print ("[TEST] Acc:", acc_test.item(), file=f)
            print ("[TEST] F1-macro:", f1_all_test.mean().item(), file=f)
            print ("[TEST] F1-scores:", f1_all_test, file=f) # print test pre-trained model
        except:
            pass
        """
    
        for nl in n_layers: # 1 or 2
            for dim in dim_features: # 128 o 256
                print ("\n\n================================================")
                print ("\nTRAINING MODELS SETTING #LAYERS:", nl, " HIDDEN DIM:", dim)
                print ("\nTRAINING MODELS SETTING #LAYERS:", nl, " HIDDEN DIM:", dim, file=f)
                print ("================================================")
                
                acc_tests=[]
                f1_tests=[]
                f1ma_tests=[]
                all_train_times=[]
                all_full_times=[]

                for i in range(num_runs):  
                    
                    if type_model=="GAT":
                        model = GAT_model(dataset.num_node_features, dim, num_classes, nl, lr, dropout=dropout, class_weights=calculated_cw)    
                    #
                    # elif type_model=="GCN":
                    #     model = GCN_model(dataset.num_node_features, dim, num_classes, nl, lr, dropout=dropout, class_weights=calculated_cw)
                    #
                    # else:
                    #     print ("Type of GNN model not supported: No GNN was intended")
                    #     return
                    
                    early_stop_callback = EarlyStopping(monitor="Val_f1-ma", mode="max", verbose=True, **config_file["early_args"])

                    path_for_savings = path_models+type_model+graph_construction # if config_file["baseline"] else path_models+type_model

                    if config_file["normalized"]:
                        pass
                        # checkpoint_callback = ModelCheckpoint(monitor="Val_f1-ma", mode="max", save_top_k=1, dirpath=path_for_savings,
                        #                                       filename=type_model+"_"+str(nl)+"L_"+str(dim)+"U_"+graph_construction+"_norm_run"+str(i)+"-OUT-{epoch:02d}-{Val_f1-ma:.2f}")
                        # wandb_logger = WandbLogger(name=model_name+'2'+type_model+"_"+str(nl)+"L_"+str(dim)+"U_"+graph_construction+"_norm_run"+str(i), save_dir=path_for_savings+"_norm", project=project_name)
                        #

                    else:
                        checkpoint_callback = ModelCheckpoint(monitor="Val_f1-ma", mode="max", save_top_k=1, dirpath=path_for_savings, 
                                                              filename=type_model+"_"+str(nl)+"L_"+str(dim)+"U_"+graph_construction+"_run"+str(i)+"-OUT-{epoch:02d}-{Val_f1-ma:.2f}")
                        wandb_logger = WandbLogger(name=model_name+'2'+type_model+"_"+str(nl)+"L_"+str(dim)+"U_"+graph_construction+"_run"+str(i), save_dir=path_for_savings, project=project_name)

                    trainer = pl.Trainer(accelerator='gpu', devices=1, callbacks=[early_stop_callback, checkpoint_callback], logger=wandb_logger, 
                                         **config_file["trainer_args"])

                    train_loader, val_loader, test_loader = partitions(dataset, dataset_test, bs=config_file["batch_size"])
                    starti = time.time()
                    
                    trainer.fit(model, train_loader, val_loader)

                    # TODO: debug because val-f1-ma = 1.00

                    print ("\n----------- Run #"+str(i)+"-----------\n", file=f)   
                    print ("\nTraining stopped on epoch:", trainer.callbacks[0].stopped_epoch, file=f)
                    train_time = time.time()-starti
                    print(f"Training time: {train_time:.2f} secs", file=f)   

                    # load best checkpoint
                    print ("Best model path:", trainer.checkpoint_callback.best_model_path, file=f)
                    print ("Best model path:", trainer.checkpoint_callback.best_model_path)
                    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
                    model.load_state_dict(checkpoint['state_dict']) 

                    preds, trues = model.predict(test_loader, cpu_store=False)

                    acc=(torch.Tensor(trues) ==preds).float().mean() 
                    f1_score = F1Score(task='multiclass', num_classes=num_classes, average='none')
                    f1_all = f1_score(preds.int(), torch.Tensor(trues).int())
                    print ("Acc Test:", acc, file=f)
                    print ("F1-macro Test:", f1_all.mean(), file=f)
                    print ("F1 for each class:", f1_all, file=f)

                    endi = time.time()
                    total_timei = endi - starti
                    print("Training + Testing time: "+ str(total_timei), file=f)
                
                    acc_tests.append(acc.cpu().numpy())
                    f1_tests.append(f1_all.cpu().numpy())
                    f1ma_tests.append(f1_all.mean().cpu().numpy())
                    all_train_times.append(train_time)
                    all_full_times.append(total_timei)


                print ("\n************************************************", file=f)
                print ("RESULTS FOR N_LAYERS:", nl, " HIDDEN DIM_FEATURES:", dim, file=f)    
                print ("Test Acc: %.3f"% np.mean(np.asarray(acc_tests)), "-- std: %.3f" % np.std(np.asarray(acc_tests)), file=f)
                print ("Test F1-macro: %.3f"%np.mean(np.asarray(f1ma_tests)), "-- std: %.3f" % np.std(np.asarray(f1ma_tests)), file=f)
                print ("Test F1 per class:", np.mean(np.asarray(f1_tests), axis=0), file=f)
                print ("Training time: %.2f"%np.mean(np.asarray(all_train_times)), "-- std: %.2f" % np.std(np.asarray(all_train_times)), file=f)
                print ("Total time: %.2f"%np.mean(np.asarray(all_full_times)), "-- std: %.2f" % np.std(np.asarray(all_full_times)), file=f)
                print ("************************************************\n\n", file=f)

        
        f.close()

    end = time.time()
    total_time = end - start
    print("\nRUNNING TIME FOR ALL THE EXPERIMENTS: "+ str(total_time))
    


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--settings_file",
        "-s",
        action="store",
        dest="settings_file",
        required=True,
        type=str,
        help="path of the settings file",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)

    main_run(config_file, args.settings_file)