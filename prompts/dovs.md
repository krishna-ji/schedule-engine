I am bothered much by Session Block Clustering problem by GA. I want you to help me solve it. 

I want array of quanta_id represetning multiple sessions for that course block . 
How can i utilize it or should i employ other techniques to enfore the session continuity within a single seesion quantas. 
e.g. a 5 load subject single session must be (x, x+1) and (y,y+1)  and (z)
multiple sessions can be there be but : wihtin a single session : the ssubsession must be continuous.
for theory subject:

if course block has odd number of load requirement: then one of the session will have 1 quanta, and all other session will have 2 quanta each. [Those quana must be continuous, the quanta inside a single session must be continuous]

if the course block has even number of load requirement: then all sessions will have 2 quanta each.

for practical subjects:
the load (P) must be scheduled in continuous quanta. 

How can i implement this logic in algorithmic based intializer? without having to design hard constraint / soft constraint for this enforcement. 


Make a full plan for migration? what is the best way possible? such that the session continutiy is not to be optimized by ga. and rather enforced auto by intializer?
create a report inside docs/plans/subsession_continuity_enforcement.md