content = open('app/services/prompt_builder.py','r',encoding='utf-8').read()
content = content.replace(
    'model_path = Path(r"E:\\courses\\data science all\\NLP_\\Arabic_News_LLM\\model")',
    'model_path = Path("./tokenizer_cache")'
)
open('app/services/prompt_builder.py','w',encoding='utf-8').write(content)
print('Done')
