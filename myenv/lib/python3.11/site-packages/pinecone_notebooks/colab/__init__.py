try:
  from IPython.display import HTML, display
  from google.colab import output
except:
  raise ImportError('This module is meant to be used in Google Colab.')
else:
  import os
  from typing import Optional

  def SetApiKey(val):
    os.environ['PINECONE_API_KEY'] = val

  def Authenticate(source_tag: Optional[str] = None):
    output.register_callback('pinecone.SetApiKey', SetApiKey)
    display(HTML(data='<script type="text/javascript" src="https://connect.pinecone.io/embed.js"></script>'))
    output.eval_js('const x = connectToPinecone((val) => google.colab.kernel.invokeFunction("pinecone.SetApiKey", [val], {}), {integrationId: "colab"})')
