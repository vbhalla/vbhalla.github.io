---
published: false
---
## Why standing up a data science function at later stage startups usually fails.

OK, so a company just raised a large round focused on becoming more tech enabled and AI with data science being core to our future growth.  We’ve heard these stories so many times, frequently ending in failure.  The below is a framework to help get started on the right foot.

**What do you expect DS to do?  **

First step is to evaluate what you currently have in place and what you expect data science to do?  As it’s fairly common to assume you need data science, but actually you need data management/platforming and BI/analytics.

Why?  If the data is poorly structured, unstable, then 95% of the work will be messy non-scalable data reconciliation and ad-hoc patches/fixes. 

A data science team can be instrumental in helping transition to a state of data stability and moving towards being a data driven organization, but clarity on the state of data is crucial PRIOR to building out a function to ensure expectations and resources can be aligned to ensure this initiative is set up for success.  

Building functions to support data management and analytics is a post in itself.  Now let’s say that your data org is in a good place, and you clearly state that you want your data science function to leverage DS to create production quality products.  Production quality can be defined as requiring reproducibility and scalability; rather than insights which can be one-off. 

**What’s needed for a strong foundation?**
- Core DS Workflow Capabilities

- Build out DS skills creating a well rounded DS team

- **Establish the DS team - org model and ownership**


**Core DS Workflow**
WorkflowImage

Process 
Documentation and workflow standards across business narrative (problem statements) down to the technical nitty gritty; at startups simplicity and a doc framework that is centralized is important.  For example, N miscellaneous powerpoints shouldn’t serve as documentation and DS teams keep powerpoint to a minimum.

Architecture
You need a ML stack that can support 3 basic functions (many ML tools/platforms provide more than 1 of these functions).  Quite simply these are:

Feature Store - how we enrich and transform both internal and external 3rd party data

Model Store - how we build and deploy DS products/models

Monitoring Store - how we monitor and improve those products/models

**Core DS Skill Sets**
SkillSetImage

**Where does data science fit in the current org?**
The above are table stakes, the number 1 reason cause of failure are alignment of incentives and establishing clear ownership.

PuzzleImage

**Data Science Operating Models**
It’s extremely important to commit to an operating model to ensure incentives are aligned, this of course can be revisited as the business changes; stakeholder needs and/or resource planning.  

A basketball metaphor probably illustrates this best.
BullsImage

Standing up a DS function at later stage startups often fail primarily because of ownership and incentives; choosing a model can help facilitate this crucial conversation with executive leadership.

Not choosing a model often results in one of the following 1 of two failed outcomes:
FlowChart

Image of Dave and Busters vs. Academic Fail

Why Centralize?
Pros: 
Global  - Bring data from all parts of company enabling a broader understanding of customer experience
Streamlined Management - career paths, peers, mentorship, collaboration, and recruiting
Accountability to DS - incentivizing resource allocation to include foundational DS 
Scale - Uniform best practices, platform, and tools across lines of business, executives, and products
Early Cons:
Quality over Speed
Difficult for managing business lines of at very different stages and needs

Why Embed?
Pros:
Local – customized to accommodate needs for business lines in different stages of development
Accountability to business line - incentivizes resource allocation focused on unique needs of individual business line
Cons:
Speed over quality 
Complex Management - career paths, peers, mentorship, collaboration, and recruiting
Doesn’t Scale - lacking standardization across all lines of businesses

Recap: 
Table Image

