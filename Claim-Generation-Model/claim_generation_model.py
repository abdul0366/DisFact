import pandas as pd
from simplet5 import SimpleT5
from sklearn.model_selection import train_test_split

# Load the CSV file into a DataFrame
df = pd.read_csv("fever_train_cleaned_processed.csv")

# Filter the DataFrame to include only rows where the value in the "Label" column is "SUPPORTS"
refute_df = df[df['Label'] == 'REFUTES']

# Write the filtered DataFrame to a new CSV file
refute_df.to_csv("refute_fever.csv", index=False) #change for support class when training for supports

path = "refute_fever.csv"
df = pd.read_csv(path, usecols = ['Claim','Evidence'])

# simpleT5 expects dataframe to have 2 columns: "source_text" and "target_text"
df = df.rename(columns={"Claim":"target_text", "Evidence":"source_text"})
df = df[['source_text', 'target_text']]



# T5 model expects a task related prefix: since it is a summarization task, we will add a prefix "summarize: "
# df['source_text'] = "summarize: " + df['source_text']

train_df, test_df = train_test_split(df, test_size=0.1) # change split as needed
train_df.shape, test_df.shape

# from simplet5 import SimpleT5

refuteModel = SimpleT5()
refuteModel.from_pretrained(model_type="t5", model_name="t5-base")
refuteModel.train(train_df=train_df,
            eval_df=test_df,
            source_max_token_len=200,
            target_max_token_len=50,
            batch_size=64, max_epochs=50, use_gpu=True)

refuteModel.load_model("t5",".../outputs/simplet5-epoch-49-train", use_gpu=True)

sample_text_to_summarize="""summarize: Rahul Gandhi has replied to Goa CM Manohar Parrikar's letter,
which accused the Congress President of using his "visit to an ailing man for political gains".
"He's under immense pressure from the PM after our meeting and needs to demonstrate his loyalty by attacking me,"
Gandhi wrote in his letter. Parrikar had clarified he didn't discuss Rafale deal with Rahul.
"""
refuteModel.predict(sample_text_to_summarize)

# Define a function to summarize the text using the model
def genRefuteClaim(text):
    return refuteModel.predict(text)[0]  # [0] to get the first element from the prediction list

# apply the refuted claim generation model to the FEMA data set



#path = "all_fema_press_releases_with_source.csv"
path = "final_corpus.csv"
df = pd.read_csv(path)

cleaned_test = df['full_text'] #.head(25)

# Apply the summarize_text function to each row in the 'Evidence' column
fema_claims = cleaned_test.apply(genRefuteClaim)

# Create a new DataFrame with the summaries
claim_df = pd.DataFrame({'Full Text': cleaned_test, 'Claim': fema_claims})


output_path = "final_corpus_refuted_claims.csv"
claim_df.to_csv(output_path, index=False)

claim_df