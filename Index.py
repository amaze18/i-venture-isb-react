import nest_asyncio
nest_asyncio.apply()
from github import Github
import boto3
import os
import openai
openai.api_key=os.environ['SECRET_TOKEN']
from llama_index.core import SimpleDirectoryReader
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

s3 = boto3.client('s3')
# github_token = os.environ['GITHUB_TOKEN']
# github_repo = os.environ['GITHUB_REPO']

session = boto3.Session(
    aws_access_key_id=os.environ['ACCESS_ID'],
    aws_secret_access_key=os.environ['ACCESS_KEY'],
)
s3 = session.client('s3')

bucket_name = 'coursechat'  # Replace with your actual S3 bucket name
# Option 1: Using list_objects with SortOrder and Prefix (if applicable)
# This approach works if you want to download the latest object from a specific prefix

# Specify a prefix to narrow down objects (optional)
# List objects in the bucket
response = s3.list_objects_v2(Bucket=bucket_name)

# Check if there are any objects
if 'Contents' not in response:
    print("No objects found in the bucket.")
else:
    # Sort objects based on last modified time
    objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)

    # Get the key of the first object (latest upload)
    latest_object_key = objects[0]['Key']

    # Specify the desired download path
    download_path = '/home/ubuntu'  # Replace with your desired path

    local_filename = f"{download_path}/{latest_object_key}"
    s3.download_file(bucket_name, latest_object_key, local_filename)


def indexgenerator(indexPath, documentsPath):

    # check if storage already exists
    if not os.path.exists(indexPath):
        print("Not existing")
        # load the documents and create the index

        entity_extractor = EntityExtractor(prediction_threshold=0.2,label_entities=False, device="cpu") # set device to "cuda" if gpu exists

        node_parser = SentenceSplitter(chunk_overlap=102,chunk_size=1024)

        transformations = [node_parser, entity_extractor]

        documents = SimpleDirectoryReader(input_dir=documentsPath).load_data()

        pipeline = IngestionPipeline(transformations=transformations)

        nodes = pipeline.run(documents=documents)

        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0),embed_model=embed_model)

        index = VectorStoreIndex(nodes, service_context=service_context)

        # store it for later
        index.storage_context.persist(indexPath)
    else:
        #load existing index
        print("Existing")
        storage_context = StorageContext.from_defaults(persist_dir=indexPath)
        index = load_index_from_storage(storage_context,service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0),embed_model=embed_model))

    return index




indexPath=f"/home/ubuntu/goutham"
documentsPath=f"/home/ubuntu"
indexgenerator(indexPath,documentsPath)

# def push_to_github(file_path):
#     # Authenticate with GitHub
#     g = Github(github_token)
#     repo = g.get_repo(github_repo)

#     # Push the file to the repository
#     with open(file_path, 'r') as file:
#         content = file.read()
#         repo.create_file(file_path, "Commit message", content)


token = os.environ['GITHUB_TOKEN']
# Repository information
repo_owner = "amaze18"
repo_name = "course_chat"

# Directory to be pushed
directory_path = indexPath

# Branch name
branch_name = "index"

def push_directory_to_github(directory_path, repo_owner, repo_name, token, branch_name):
    # Authenticate to GitHub using token
    g = Github(token)

    # Get the repository
    repo = g.get_user(repo_owner).get_repo(repo_name)

    # Create branch if not exists
    branches = repo.get_branches()
    branch_exists = False
    for b in branches:
        if b.name == branch_name:
            branch_exists = True
            break
    if not branch_exists:
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=repo.get_branch("main").commit.sha)

    # Create a folder in the repository with the directory name
    dir_name = os.path.basename(directory_path)
    repo.create_file(f"{dir_name}/.gitkeep", f"Creating {dir_name} folder", '')

    # Push all files in the directory to the repository
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'rb') as file:  # Open the file in binary mode
            content = file.read()  # Read content as bytes
            # Convert bytes to UTF-8 encoded string
            content_utf8 = content.decode('utf-8', 'ignore')
            repo.create_file(f"{dir_name}/{file_name}", f"Add {file_name}", content_utf8, branch=branch_name)

    print("Directory pushed successfully to GitHub.")

# Push the directory to GitHub
push_directory_to_github(directory_path, repo_owner, repo_name, token,branch_name)
