"""This module contains argument bots. 
These agents should be able to handle a wide variety of topics and opponents.
They will be evaluated using methods in `evaluate.py`.
I've included a few to get your started."""

import logging
from rich.logging import RichHandler
from pathlib import Path
import random
import glob
from collections import deque
from dialogue import Dialogue
from agents import Agent, ConstantAgent, LLMAgent
from kialo import Kialo

# Use the same logger as agents.py, since argubots are agents;
# we split this file 
# You can change the logging level there.
log = logging.getLogger("agents")    

#############################
## Define some basic argubots
#############################

# Airhead (aka Absentia or Acephalic) always says the same thing.

airhead = ConstantAgent("Airhead", "I know right???")

# Alice is a basic prompted LLM.  You are trying to improve on Alice.
# Don't change the prompt -- instead, make a new argubot with a new prompt.

alice = LLMAgent("Alice",
                 system="You are an intelligent bot who wants to broaden your user's mind. "
                        "Ask a conversation starter question.  Then, WHATEVER "
                        "position the user initially takes, push back on it. "
                        "Try to help the user see the other side of the issue. "
                        "Answer in 1-2 sentences. Be thoughtful and polite.")

############################################################
## Other argubot classes and instances -- add your own here! 
############################################################

class KialoAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            prev_turn = d[-1]['content']  # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(prev_turn, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
            
            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])
        
        return claim    
    
  
akiko = KialoAgent("Akiko", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files


###########################################
# Define your own additional argubots here!
###########################################

class RAGAgent(LLMAgent):
    """RAGAgent combines the power of LLM with retrieval from knowledge base."""
    
    def __init__(self, name: str, kialo: Kialo):
        base_prompt = ("You are an intelligent debater who uses evidence and logic to engage in thoughtful discussion. "
                      "When presented with arguments from past debates, incorporate their insights while maintaining a balanced perspective. "
                      "Your goal is to help others see multiple sides of an issue and broaden their thinking. "
                      "First understand their position, then respond with well-reasoned points that encourage deeper consideration. "
                      "Be articulate but respectful, and keep responses focused and concise.")
        super().__init__(name, system=base_prompt)
        self.kialo = kialo
        
    def get_context(self, d: Dialogue) -> str | None:
        """Helper method to get relevant context from Kialo database."""
        try:
            if len(d) == 0:
                claims = self.kialo.random_chain(n=2)
                return f"Consider this chain of reasoning as inspiration (but don't repeat it directly): {'; '.join(claims)}"
            
            prev_turn = d[-1]['content']
            similar_claims = self.kialo.closest_claims(prev_turn, n=3)
            if not similar_claims:
                return None
            

            relevant_info = []
            for claim in similar_claims:
                relevant_info.append(f"Claim: {claim}")
                pros = self.kialo.pros.get(claim, [])[:2]
                cons = self.kialo.cons.get(claim, [])[:2]
                relevant_info.extend([f"Supporting argument: {arg}" for arg in pros])
                relevant_info.extend([f"Counter argument: {arg}" for arg in cons])
            
            if relevant_info:
                return " | ".join(relevant_info[:4])
            return None
            
        except Exception as e:
            log.error(f"Error getting context in RAGAgent: {str(e)}")
            return None

    def response(self, d: Dialogue, **kwargs) -> str:
        """Generate a response using retrieval-augmented generation."""
        try:
            context = self.get_context(d)
            if context:
                orig_system = self.kwargs_format.get('system', '')
                self.kwargs_format['system'] = f"{orig_system}\n\nRelevant context from previous debates:\n{context}"
                try:
                    return super().response(d, **kwargs)
                finally:
                    self.kwargs_format['system'] = orig_system
            return super().response(d, **kwargs)
        except Exception as e:
            log.error(f"Error in RAGAgent response: {str(e)}")
            return super().response(d, **kwargs)


aragorn = RAGAgent("Aragorn", Kialo(glob.glob("data/*.txt")))


class ChainOfThoughtAgent(RAGAgent):
    """ChainOfThoughtAgent: Enhanced RAGAgent with Chain of Thought reasoning through private thoughts."""
    
    def __init__(self, name: str, kialo: Kialo):
        base_prompt = (
            "You are a thoughtful dialogue facilitator using chain-of-thought reasoning to better understand and respond to others. "
            "Before responding, you will first analyze the situation in a private thought that considers:"
            "\n1. The speaker's current emotional state and possible underlying concerns"
            "\n2. What they might be avoiding or hesitant to discuss"
            "\n3. How to make them feel more comfortable opening up"
            "\n4. A strategy to help broaden their perspective"
            "\nAfter your private analysis, provide a response that implements your chosen strategy."
            "\nKeep responses concise (2-3 sentences) and maintain a warm, respectful tone."
        )
        super().__init__(name=name, kialo=kialo)
        self.kwargs_format['system'] = base_prompt
        
    def get_private_thought(self, d: Dialogue) -> str:
        """Generate a private thought analyzing the current dialogue state."""
        if len(d) == 0:
            return "This is the start of our conversation. I should begin with an engaging but non-threatening topic to establish rapport."
            

        thought_prompt = ("Analyze the conversation so far and generate a private thought that considers:\n"
                         "1. What is the speaker's apparent position and emotional state?\n"
                         "2. What might they be avoiding or hesitant to discuss?\n"
                         "3. How can we make them more comfortable opening up?\n"
                         "4. What strategy would work best to broaden their perspective?\n")
        

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": thought_prompt},
                {"role": "user", "content": str(d)}
            ],
            model=self.model,
            temperature=0.7
        )
        return response.choices[0].message.content

    def response(self, d: Dialogue, **kwargs) -> str:
        """Generate a response using chain-of-thought reasoning."""
        try:
            private_thought = self.get_private_thought(d)
            log.info(f"[black on bright_yellow]Victor's private thought:\n{private_thought}[/black on bright_yellow]")
            
            orig_system = self.kwargs_format.get('system', '')
            
            self.kwargs_format['system'] = (f"{orig_system}\n\n"
                                          f"Your private analysis of the situation:\n{private_thought}\n\n"
                                          f"Based on this analysis, respond in a way that implements your strategy.")
            try:
                return super().response(d, **kwargs)
            finally:
                self.kwargs_format['system'] = orig_system
                
        except Exception as e:
            log.error(f"Error in Victor2 response: {str(e)}")
            return super().response(d, **kwargs)


chain_of_thought = ChainOfThoughtAgent("ChainOfThought", Kialo(glob.glob("data/*.txt")))

class Victor(RAGAgent):
    """Victor: A simple chain-of-thought enhanced RAGAgent that thinks before responding."""
    
    def __init__(self, name: str, kialo: Kialo):
        base_prompt = (
            "You are a thoughtful and intelligent debater who first analyzes the situation privately before responding. "
            "Consider the context and the speaker's position very carefully. "
            "Keep responses focused and respectful, aiming to broaden understanding."
        )
        super().__init__(name=name, kialo=kialo)
        self.kwargs_format['system'] = base_prompt

    def response(self, d: Dialogue, **kwargs) -> str:
        """Generate a response using simple chain-of-thought reasoning."""
        try:
            context = self.get_context(d)
            
            thought_prompt = "Given the context and dialogue, what's the key point to address?"
            thought = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": thought_prompt},
                    {"role": "user", "content": 
                     f"context: {str(context)}"
                     f"\n\nDialogue so far: {str(d)}"
                     }
                ],
                model=self.model,
                temperature=0.4
            ).choices[0].message.content
            
            log.info(f"[black on bright_yellow]Victor's thought:\n{thought}[/black on bright_yellow]")
            
            orig_system = self.kwargs_format['system']
            context_str = f"\n\nRelevant context: {context}" if context else ""
            self.kwargs_format['system'] = f"{orig_system}\n\nYour analysis: {thought}{context_str}"
            
            try:
                return super().response(d, **kwargs)
            finally:
                self.kwargs_format['system'] = orig_system
                
        except Exception as e:
            log.error(f"Error in Victor response: {str(e)}")
            return super().response(d, **kwargs)


victor = Victor("Victor", Kialo(glob.glob("data/*.txt")))

class FewShotLearningAgent(RAGAgent):
    """FewShotLearningAgent: Enhanced RAGAgent that uses few-shot prompting for better query formation."""
    
    def __init__(self, name: str, kialo: Kialo):
        base_prompt = ("You are an intelligent debater who uses evidence and logic to engage in thoughtful discussion. "
                      "Your goal is to help others see multiple sides of an issue and broaden their thinking.")
        super().__init__(name=name, kialo=kialo)
        self.kwargs_format['system'] = base_prompt
        
        self.query_examples = [
            {"role": "system", "content": "Transform casual dialogue responses into clear, formal claims suitable for debate."},
            {"role": "user", "content": "Sounds fishy that they developed the vaccine so fast."},
            {"role": "assistant", "content": "A vaccine developed very quickly cannot be trusted. Rapid development suggests inadequate safety testing and potential risks."},
            {"role": "user", "content": "I just don't think kids should be forced to be vegan."},
            {"role": "assistant", "content": "Enforcing a vegan diet on children is ethically wrong and potentially harmful to their development."},
            {"role": "user", "content": "Biden is clearly senile and unfit for office."},
            {"role": "assistant", "content": "Joe Biden's cognitive abilities and mental state make him unqualified to serve as President of the United States."}
        ]

    def reformulate_query(self, dialogue_turn: str) -> str:
        """Transform a casual dialogue turn into a formal claim using few-shot prompting."""
        messages = self.query_examples.copy()
        messages.append({"role": "user", "content": dialogue_turn})
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            log.error(f"Error in query reformulation: {str(e)}")
            return dialogue_turn

    def get_context(self, d: Dialogue) -> str | None:
        """Enhanced context retrieval with better query formation."""
        try:
            if len(d) == 0:
                claims = self.kialo.random_chain(n=2)
                return f"Consider this chain of reasoning as inspiration (but don't repeat it directly): {'; '.join(claims)}"
            
            reformulated_query = self.reformulate_query(d[-1]['content'])
            
            similar_claims = self.kialo.closest_claims(reformulated_query, n=3)
            if not similar_claims:
                return None
            
            relevant_info = []
            for claim in similar_claims:
                relevant_info.append(f"Claim: {claim}")
                pros = self.kialo.pros.get(claim, [])[:2]
                cons = self.kialo.cons.get(claim, [])[:2]
                relevant_info.extend([f"Supporting argument: {arg}" for arg in pros])
                relevant_info.extend([f"Counter argument: {arg}" for arg in cons])
            
            if relevant_info:
                return " | ".join(relevant_info[:4])
            return None
            
        except Exception as e:
            log.error(f"Error getting context in Victor3: {str(e)}")
            return None

few_shot_learning = FewShotLearningAgent("FewShotLearning", Kialo(glob.glob("data/*.txt")))


class StructuredDialogueAgent(RAGAgent):
    """StructuredDialogueAgent: Enhanced RAGAgent with improved prompt engineering for better dialogue engagement.
    Uses LAEB (Listen, Acknowledge, Explore, Bridge) framework."""
    
    def __init__(self, name: str, kialo: Kialo):
        base_prompt = (
            "You are a highly skilled dialogue facilitator and critical thinking expert. Your approach is:"
            "\n1. LISTEN: Carefully analyze the speaker's position, emotional state, and underlying assumptions."
            "\n2. ACKNOWLEDGE: Show genuine understanding of their perspective before presenting alternatives."
            "\n3. EXPLORE: Use the Socratic method to help them examine their beliefs more deeply."
            "\n4. INFORM: Share relevant evidence and counterarguments in a non-threatening way."
            "\n5. BRIDGE: Find common ground and show how different viewpoints can coexist."
            "\nYour goals are to:"
            "\n- Foster intellectual curiosity and open-mindedness"
            "\n- Help people recognize the complexity of issues"
            "\n- Encourage evidence-based thinking"
            "\n- Maintain emotional safety in challenging discussions"
            "\n- Build bridges between different perspectives"
            "\nKeep responses concise (2-3 sentences) but impactful. Use a tone that is warm, respectful, and intellectually engaging."
            "\nWhen using evidence from past debates, present it as an interesting perspective to consider rather than absolute truth."
        )
        super().__init__(name=name, kialo=kialo)
        self.kwargs_format['system'] = base_prompt

prompt_engineering = StructuredDialogueAgent("StructuredDialogue", Kialo(glob.glob("data/*.txt")))

