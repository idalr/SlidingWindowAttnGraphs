import xml.etree.ElementTree as ET
import pandas as pd
import re

## Data ZENODO HND --- https://zenodo.org/records/5776081

def clean_str(string): 
    string = re.sub(r"\_", " ", string)
    string = re.sub(r"\-", " ", string) 
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\'", " ", string)
    return string


def load_data(in_path, data_train, labels_train, data_test, labels_test, with_val=False):
    ### check if there is a folder with processed data and load from there. Otherwise, create the dfs and save
    try:
        #load
        if with_val:
            print("Loading from Processed folder")
            df_train = pd.read_csv(in_path+"/Processed/df_train.csv")
            try:
                df_val = pd.read_csv(in_path+"/Processed/df_validation.csv")
            except:
                df_val = pd.read_csv(in_path+"/Processed/df_val.csv")
            df_test = pd.read_csv(in_path+"/Processed/df_test.csv")
            return df_train, df_val, df_test
        else:
            print("Loading from Processed folder")
            df_train = pd.read_csv(in_path+"/Processed/df_train.csv")
            df_test = pd.read_csv(in_path+"/Processed/df_test.csv")
            return df_train, df_test

    except:
        print("Processed data not found.")
        if with_val:
            print("Please check if the data is in the correct folder or if the data has been processed before.")
            return None, None, None
        else:
            print ("Creating dataframes...") ### for HND dataset
            df_train = create_dataframe(in_path, data_train, labels_train)
            df_test = create_dataframe(in_path, data_test, labels_test)

            # Save dataframes
            df_train.to_csv(in_path+"/Processed/df_train.csv",index=False)
            df_test.to_csv(in_path+"/Processed/df_test.csv",index=False)

            return df_train, df_test


def create_dataframe(in_path, data_df, labels_df): ### from XML to DataFrame (HND dataset)

    tree_df = ET.parse(in_path+data_df)
    root_df = tree_df.getroot()

    tree_labels = ET.parse(in_path+labels_df)
    root_labels = tree_labels.getroot()
    
    lab_dict={"true":1, "false":0, "True":1, "False":0}
    df_retrieve= pd.DataFrame(columns=["article_id", "article_title", "article_text", "article_label"])
    dict_text_file={}

    for child in root_df:        
        article_id= int(child.attrib['id'])
        article_title = clean_str(child.attrib['title'])
        article_text= ""

        for subchild in child:
            article_subtext = subchild.text
            if article_subtext!=None:
                article_subtext= " "+article_subtext
                article_subtext= clean_str(article_subtext)
                article_text+= article_subtext
            
        dict_text_file[article_id]=(article_title, article_text)
        
    dict_lab_file={}            
    for child_label in root_labels:
        article_id_lab = int(child_label.attrib['id'])
        article_label= lab_dict[child_label.attrib['hyperpartisan']]
        dict_lab_file[article_id_lab]= article_label
    
    for article_id in dict_text_file.keys() and dict_lab_file.keys():
        if dict_text_file[article_id][1].strip()!="":
            df_retrieve.loc[len(df_retrieve)] = {"article_id":article_id, "article_title":dict_text_file[article_id][0], "article_text":dict_text_file[article_id][1], "article_label":dict_lab_file[article_id]}
        else:
            print ("Article ID", article_id, "is empty -- Ignoring sample")

        
    return df_retrieve 