FROM apache/spark:3.5.1

USER root
RUN pip install --no-cache-dir \
    delta-spark==3.1.0 \
    great-expectations \
    pandas numpy faker networkx \
    scikit-learn xgboost shap pyyaml

USER 185