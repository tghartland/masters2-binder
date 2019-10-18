import argparse
import json

import redis

parser = argparse.ArgumentParser()
parser.add_argument("--particle", help="particle name", required=True)
parser.add_argument("--workflow", help="workflow name", required=True)
parser.add_argument("--sim-file", help="file with simulated data to use", required=True)
parser.add_argument("--sim-hist", help="histogram to use from simulated data file", required=True)
parser.add_argument("--fb", type=int, help="number of inverse femtobarns of data", required=True)
parser.add_argument("--masses", help="list of masses", required=True)

parser.add_argument("--redis-host", help="redis hostname", required=True)
parser.add_argument("--redis-port", help="redis port", required=True)

args = parser.parse_args()

r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)

r.hset(args.workflow, "workflow", args.workflow)
r.hset(args.workflow, "particle", args.particle)
r.hset(args.workflow, "sim-file", args.sim_file)
r.hset(args.workflow, "sim-hist", args.sim_hist)
r.hset(args.workflow, "fb", args.fb)

r.rpush("{}:mass-points".format(args.workflow), *json.loads(args.masses))
