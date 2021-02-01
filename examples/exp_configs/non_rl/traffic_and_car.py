# import Flow's base network class
from flow.networks import Network

# define the network class, and inherit properties from the base network class
class myNetwork(Network):
    pass

ADDITIONAL_NET_PARAMS = {
    "radius": 40,
    "num_lanes": 1,
    "speed_limit": 30,
}

class myNetwork(myNetwork):  # update my network class

    def specify_nodes(self, net_params):
        # one of the elements net_params will need is a "radius" value
        r = net_params.additional_params["radius"]

        # specify the name and position (x,y) of each node
        nodes = [{"id": "top1", "x": 0,  "y": 0},
                 {"id": "top2",  "x": 0,  "y": r},
                 {"id": "top3",    "x": 0,  "y": 2*r},
                 {"id": "bot1",   "x": r, "y": 0},
                 {"id": "bot2",   "x": r, "y": r},
                 {"id": "bot3",   "x": r, "y": 2*r}]
                 
    def specify_edges(self, net_params):
        r = net_params.additional_params["radius"]
        edgelen = r * pi / 2
        # this will let us control the number of lanes in the network
        lanes = net_params.additional_params["num_lanes"]
        # speed limit of vehicles in the network
        speed_limit = net_params.additional_params["speed_limit"]

        edges = [
            {
                "id": "edge0",
                "numLanes": lanes,
                "speed": speed_limit,     
                "from": "bottom", 
                "to": "right", 
                "length": edgelen,
                "shape": [(r*cos(t), r*sin(t)) for t in linspace(-pi/2, 0, 40)]
            },
            {
                "id": "edge1",
                "numLanes": lanes, 
                "speed": speed_limit,
                "from": "right",
                "to": "top",
                "length": edgelen,
                "shape": [(r*cos(t), r*sin(t)) for t in linspace(0, pi/2, 40)]
            },
            {
                "id": "edge2",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "top",
                "to": "left", 
                "length": edgelen,
                "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi/2, pi, 40)]},
            {
                "id": "edge3", 
                "numLanes": lanes, 
                "speed": speed_limit,
                "from": "left", 
                "to": "bottom", 
                "length": edgelen,
                "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            }
        ]
        return nodes