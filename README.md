# indoorCont
ABM simulation for indoor contact


## License
indoorContact / version 0.1.0
- install:

```python
!pip install indoorContact
```


### 1. Purpose

###### The purpose of this model is to observe the patterns of individual spatiotemporal contact in indoor space, which are shaped by the interactions between humans (agents) and their environment (e.g., other agents and indoor spatial structures).

### 2. Simulation process


<p align="center">
<img src="/indoorContact/screenshot/Fig1.png" alt="Overview of simulation framework" width="700"/>
</p>
<p align="center">
<h>Figure 1. Overview of Simulation Framework</h>
</p>

###### The simulation process, as depicted in Figure 1, begins with the creation of an indoor space, followed by the initialization of parameters. The indoor space is determined by specifying the width and height in meters, with the option to add a specified number of obstacles. These obstacles, each measuring 1m by 1m, are randomly deployed within the space. Entrances are created based on the specified number and coordinates, allowing agents to enter and exit at these positions. Furthermore, we can import the indoor space in a grid format that includes shape, obstacle placement and size, and entrance specifications, which offers flexibility in simulating various spatial configurations for users.
###### Following the creation of the indoor space, global parameters, such as group proportion, population, and simulation duration, are set. Subsequently, 'n' agents are generated, each endowed with individual attributes like speed and activeness. The simulation starts when the first agent enters the space, setting the beginning of the simulation time. Concurrently, a time-stamped table is generated, recording attributes such as agent position, group association, and contact count for each time unit. This data can be visualized in a movement animation, illustrating how individuals form groups and move within the indoor space over time.


### 3. Agent's path selection avoiding obstacles

Algorithm 1. Path selection avoiding obstacles

<img src="/indoorContact/screenshot/Algorithm1.png" alt="Path selection avoiding obstacles" width="500"/>


###### The algorithm emphasizes a straight route to a designated destination. If obstacles block this direct path, the agent chooses one of eight directions closest to the target. By repetitively applying this method, the algorithm avoids barriers and finds an optimal or nearly optimal route.
###### Specifically, the agent's main action is 'Check Path'. Here, it ascertains if any barriers \(O\) lie between its current spot \(P_{current}\) and the goal \(P_{dest}\). If no obstructions exist, it indicates a straight path. When there's a direct route, the vector \(\Delta P\), which showcases the agent's subsequent location when traveling at speed \(s\), turns into a unit direction vector normalized by the true distance from the current to the target spot. In essence, the agent shifts by \(\Delta P\) from its current spot, concluding the algorithm. 
###### Should obstacles block the straightforward route, the algorithm sets the minDist variable (indicating the distance to \(P_{dest}\)) to infinity. For each of the eight direction vectors \(d\), it formulates the potential direction vector \(\Delta P\) as the product of speed \(s\) and direction \(d\). If \(\Delta P\) gets the agent nearer to \(P_{dest}\) without any obstacle \(O\) interference, the optimal direction \(d_{best}\) updates to the present direction \(d\), and minDist gets updated. Post evaluating every direction, the agent advances in the \(d_{best}\) direction at speed \(s\). This loop continues until the agent arrives at \(P_{dest}\) or no viable routes remain.


### 4. Measuring contact time between agents


Equation 1. Measuring contact time between agents

  <img src="/indoorContact/screenshot/Equation2.png" alt="Measuring contact time between agents" width="450"/>


###### The equation determines the contact duration between individual <i>i</i> and other agents <i>k</i> by segmenting each trajectory into 0.1-second interval vertices <i>j</i>. It then calculates the Euclidean distance <i>d</i> between matching vertices <i>j</i>. If this distance is under 6 feet (or 1.8m), 0.1 gets added to the contact sec. By executing this for all trajectories of <i>i</i>, the duration <i>i</i> has been in contact with <i>k</i> is gauged. Figure 3 depicts this process: when two agents reside within 6 feet while traversing three vertices, the contact sec for that scenario equals 0.3.





## Usage (Run simulation and export movie clip)

### 1. add space from data or make space

``` python
import indoorContact as ic


# -- with data --
# with entrance
space, FDD = ic.makeSpace(DFMap= ic.space_no_ent, FDD = True) #entrance = 2, obstacles = 1

# without entrance
space, entrance, FDD = ic.makeSpace(DFMap= ic.space_ent, FDD = True)

# -- without data --
# no obstacles
space = ic.makeSpace(space_x = 10, space_y = 10)

# deploy obstacles
space, obstacles_loc = ic.makeSpace(space_x = 10, space_y = 10, obstacles= 10)

# with chair
space, obstacles_loc = ic.makeSpace(space_x = 10, space_y = 10, obstacles = 10, chairs = 5)

# with entrance
space, entrance, FDD, obstacles_loc = ic.makeSpace(space_x= 15, space_y = 10, obstacles= 10, FDD = True, entrance = {'x': [15,15], 'y': [0,3]}) #x [from: to] / y: [from: to]

print(space)
```


<p align="center">
<img src="/indoorContact/screenshot/space.png" alt="This space is made of 0 and 1 (1: Obstacle, 2: chair, 3: wall)" width="350"/>
</p>
<p align="center">
<h>Figure 2. This space is made of 0 and 1 (1: Obstacle, 2: chair, 3: wall)</h>
</p>

###### FDD represents the degree of disturbance caused by obstacles to people's movement, ranging between 0 and 1. A higher value signifies more obstruction to smooth movement. This is depicted in Equation 1. Here, T signifies the total indoor area, O denotes the obstacle area, P is the total passage area excluding obstacles, and <i>n</i> represents the number of passage segments. For instance, in Figure 2, with no obstacles, FDD is 0, whereas a fully obstructed space yields an FDD of 1. If there's a single passage, FDD equals 0.5, and with three passages, it's 0.833.



<p align="center">
<img src="/indoorContact/screenshot/Fig2.png" alt="FDD (Flow Disturbance Degree)" width="450"/>
</p>
<p align="center">
<h>Figure 3. FDD (Flow Disturbance Degree)</h>
</p>



### 2. run contact simulation and count contact

``` python

# no scenario
result_df = ic.contact_Simulation(speed = [0.75, 1.8], activity = 5, totalPop = 10, space = space, entrance = entrance, total_time =100)
result_df = ic.countContact(result_df)

# adding chair scenario
space, obstacles_loc = ic.makeSpace(space_x = 10, space_y = 10, obstacles = 10, chairs = 5)
result_df = ic.contact_Simulation(speed = [0.75, 1.8], activity = 5, chair_scenario = [3, 10, 20], totalPop = 10, space = space, entrance = entrance, total_time =100)

# adding group scenario
space, obstacles_loc = ic.makeSpace(space_x = 10, space_y = 10, obstacles= 10)
result_df = ic.contact_Simulation(speed = [0.75, 1.8], activity = 5, group_scenario = [0.5, [2,3], [50, 10]], totalPop = 15, space = space, total_time =100)

```


<p align="center">
<img src="/indoorContact/screenshot/result_df.head().png" alt=" result dataframe of simulation" width="750"/>
</p>
<p align="center">
<h>Figure 4. result dataframe of simulation</h>
</p>


- time: total simulation time
- ID: unique ID of agent
- Sec: each second that each agent stay for
- X, Y: coordinates of agents
- Contact_count: the number of contact
- Vertex: verteces of trajectories
- Speed: speed of agents
- sit: sit or not (chair scenario)
- exit: 1 once agent go out
- STUCK: if agent is stuck and lose the way
- totalP: total population
- Chair: chair location where the agent sit
- group: group (1) or not (0)
- groupedP: population who are in the same group
- Contact_Sec: duration (second) of contact
- Contact_IDs: ID who encounter
- Contact_Pop: population that the agent encounter (physically contact)


### 3. export movie clip of simulation

``` python

# movie clip
ic.simul_clip_export('C:/Users/', result_df, space, 'result_clip.mp4')

```




<p align="center">
<img src="/indoorContact/screenshot/contact_exper2.png" alt="Movement animation screenshot (Legend: Group proportion, Group size, Population, Obstacles)" width="600"/>
</p>
<p align="center">
<h>Figure 5. Movement animation screenshot (Legend: Group proportion, Group size, Population, Obstacles)</h>
</p>



--- Movie clip of ABM simulation ---

https://youtube.com/shorts/c2OObeFbAQg?feature=share

https://youtube.com/shorts/Jbbe1vRs6AU?feature=share

https://youtube.com/shorts/pJMyV7u3Tw4?feature=share

https://youtube.com/shorts/Rs5XBTqaasY?feature=share


---

## Related Document: 
 will be added

## Author

- **Author:** Moongi Choi
- **Email:** u1316663@utah.edu
