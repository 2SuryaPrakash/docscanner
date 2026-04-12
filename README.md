openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj '/CN=localhost'
streamlit run app.py --server.sslCertFile=cert.pem --server.sslKeyFile=key.pem
