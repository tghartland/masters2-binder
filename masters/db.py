import redis

HOST = "123.456.789.0"
PORT = 32131

def create_client():
    r = redis.Redis(host=HOST, port=PORT, db=0)
    return r
    
def get_data(client, workflow):
    data = client.hgetall(workflow)
    data = {str(k, "utf8"): str(v, "utf8") for k, v in data.items()}
    mass_points = list(map(int, client.lrange("{}:mass-points".format(workflow), 0, -1)))
    expected_limits = {m: list(map(float, client.lrange("{}:expected-limits:{}".format(workflow, m), 0, -1))) for m in mass_points}
    data_limits = {int(k): float(v) for k, v in client.hgetall("{}:data-limits".format(workflow)).items()}
    data.update({
        "mass-points": mass_points,
        "expected-limits": expected_limits,
        "data-limits": data_limits,
    })
    return data
