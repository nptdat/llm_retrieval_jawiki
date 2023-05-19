# llm_retrieval
QA system based on jawiki data using [ChatGPT retrieval pattern](https://github.com/openai/chatgpt-retrieval-plugin)


# Start qdrant

```
$ docker-compose up qdrant
```

## View qdrant
```
$ curl localhost:6333/collections
$ curl localhost:6333/collections/llm_retrieval
```

# Start indexing batch
```
$ cd llm_retrieval/src
$ poetry run indexing
```

# Start qa service
```
$ poetry run start
```


# Sample questions
- 東芝医療情報システムズはいつ設立されましたか？
  - 東芝医療情報システムズは2004年4月に設立されました。
