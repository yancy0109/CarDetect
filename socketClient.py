import websocket

#获取WebSocket客户端
def getClient(ip,port,id,enableTrace = True):
    connectUrl = "ws://"+ip+":"+port+"/websocket/"+id
    print(connectUrl)
    websocket.enableTrace(enableTrace)
    ws = websocket.WebSocket()
    ws.connect(connectUrl)
    return ws