"""
This module contains implementations of different dialogue agents for mind-opening discourse.
Based on evaluation metrics comparing Aragorn (RAGAgent) with Alice, Aragorn demonstrated superior 
performance with a total score of 23.1 vs Alice's 22.5. Due to this performance advantage, 
all subsequent Victor models (Victor1-4) use Aragorn's RAGAgent as their base implementation, 
building upon its successful retrieval-augmented generation approach while adding specific 
enhancements for different aspects of dialogue interaction.
"""

import logging
from rich.logging import RichHandler
from pathlib import Path
import random
import glob
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
    
# Akiko doesn't use an LLM, but looks up an argument in a database.
  
akiko = KialoAgent("Akiko", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files

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
                # For first turn, use random claims as inspiration
                claims = self.kialo.random_chain(n=2)
                return f"Consider this chain of reasoning as inspiration (but don't repeat it directly): {'; '.join(claims)}"
            
            # Get similar claims for the last message
            prev_turn = d[-1]['content']
            similar_claims = self.kialo.closest_claims(prev_turn, n=3)
            if not similar_claims:
                return None
            
            # Build context with both pros and cons
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
                # Store original system message
                orig_system = self.kwargs_format.get('system', '')
                # Update system message with context
                self.kwargs_format['system'] = f"{orig_system}\n\nRelevant context from previous debates:\n{context}"
                try:
                    return super().response(d, **kwargs)
                finally:
                    # Restore original system message
                    self.kwargs_format['system'] = orig_system
            return super().response(d, **kwargs)
        except Exception as e:
            log.error(f"Error in RAGAgent response: {str(e)}")
            return super().response(d, **kwargs)

# Create Aragorn instance using all available Kialo data
aragorn = RAGAgent("Aragorn", Kialo(glob.glob("data/*.txt")))

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

# Create StructuredDialogue instance
structured_dialogue = StructuredDialogueAgent("StructuredDialogue", Kialo(glob.glob("data/*.txt")))

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
            
        # Create a prompt for private thought generation
        thought_prompt = ("Analyze the conversation so far and generate a private thought that considers:\n"
                         "1. What is the speaker's apparent position and emotional state?\n"
                         "2. What might they be avoiding or hesitant to discuss?\n"
                         "3. How can we make them more comfortable opening up?\n"
                         "4. What strategy would work best to broaden their perspective?\n")
        
        # Get LLM to generate private thought using the client's create method
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
            # First generate private thought
            private_thought = self.get_private_thought(d)
            
            # Log the private thought for debugging/analysis
            log.info(f"[black on bright_yellow]Victor's private thought:\n{private_thought}[/black on bright_yellow]")
            
            # Store original system message
            orig_system = self.kwargs_format.get('system', '')
            
            # Update system message with private thought context
            self.kwargs_format['system'] = (f"{orig_system}\n\n"
                                          f"Your private analysis of the situation:\n{private_thought}\n\n"
                                          f"Based on this analysis, respond in a way that implements your strategy.")
            try:
                return super().response(d, **kwargs)
            finally:
                # Restore original system message
                self.kwargs_format['system'] = orig_system
                
        except Exception as e:
            log.error(f"Error in Victor2 response: {str(e)}")
            return super().response(d, **kwargs)

# Create ChainOfThought instance
chain_of_thought = ChainOfThoughtAgent("ChainOfThought", Kialo(glob.glob("data/*.txt")))

class FewShotLearningAgent(RAGAgent):
    """FewShotLearningAgent: Enhanced RAGAgent that uses few-shot prompting for better query formation."""
    
    def __init__(self, name: str, kialo: Kialo):
        base_prompt = ("You are an intelligent debater who uses evidence and logic to engage in thoughtful discussion. "
                      "Your goal is to help others see multiple sides of an issue and broaden their thinking.")
        super().__init__(name=name, kialo=kialo)
        self.kwargs_format['system'] = base_prompt
        
        # Few-shot examples for query formation
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
            log.info(f"[black on bright_yellow]Reformulated query: {response.choices[0].message.content}[/black on bright_yellow]")
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
            
            # Reformulate the last message into a formal claim
            reformulated_query = self.reformulate_query(d[-1]['content'])
            
            # Get similar claims using the reformulated query
            similar_claims = self.kialo.closest_claims(reformulated_query, n=3)
            if not similar_claims:
                return None
            
            # Build context with both pros and cons
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

# Create FewShotLearning instance
few_shot_learning = FewShotLearningAgent("FewShotLearning", Kialo(glob.glob("data/*.txt")))

class ParallelGenerationAgent(RAGAgent):
    """ParallelGenerationAgent: Enhanced RAGAgent with parallel generation and self-evaluation capabilities."""
    
    def __init__(self, name: str, kialo: Kialo):
        base_prompt = ("You are an intelligent debater who uses evidence and logic to engage in thoughtful discussion. "
                      "Your goal is to help others see multiple sides of an issue and broaden their thinking. "
                      "Generate thoughtful responses and evaluate them based on:\n"
                      "1. Persuasiveness - How well it addresses the user's position\n"
                      "2. Empathy - How well it acknowledges the user's perspective\n"
                      "3. Evidence - How well it uses supporting facts and logic\n"
                      "4. Engagement - How likely it is to encourage further discussion")
        super().__init__(name=name, kialo=kialo)
        self.kwargs_format['system'] = base_prompt
        
    def reformulate_queries(self, message: str, n: int = 3) -> list[str]:
        """Generate multiple reformulations of the user's message in parallel."""
        prompt = ("Transform this casual statement into a formal debate claim. "
                 "Focus on the core argument or position being expressed. "
                 "Make it clear and specific enough to match against a debate database.")
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message}
        ]
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                n=n,
                temperature=0.7
            )
            responses = [choice.message.content for choice in response.choices]
            log.info(f"[black on bright_yellow]Generated {n} query reformulations[/black on bright_yellow]")
            return responses
        except Exception as e:
            log.error(f"Error in parallel query reformulation: {str(e)}")
            return [message]  # fallback to original message

    def generate_and_evaluate_responses(self, d: Dialogue, context: str, n: int = 3) -> str:
        """Generate multiple responses and select the best one based on self-evaluation."""
        eval_prompt = (f"Generate a response to the dialogue that uses the provided context. "
                      f"Then evaluate it on a scale of 1-10 based on:\n"
                      f"- Persuasiveness (how well it addresses the core issue)\n"
                      f"- Empathy (how well it acknowledges their perspective)\n"
                      f"- Evidence (how well it uses facts and logic)\n"
                      f"- Engagement (likelihood of continuing productive discussion)\n"
                      f"Format: 'RESPONSE: <your response>\nEVALUATION: <your evaluation>\nSCORE: <just the number, nothing else>'")
        
        messages = [
            {"role": "system", "content": eval_prompt},
            {"role": "user", "content": f"Dialogue:\n{str(d)}\n\nContext:\n{context}"}
        ]
        
        try:
            # Generate multiple responses with evaluations
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                n=n,
                temperature=0.7
            )
            candidates = [choice.message.content for choice in response.choices]
            
            # Parse responses and scores
            best_response = None
            best_score = -1
            
            for candidate in candidates:
                try:
                    response_part = candidate.split("RESPONSE:")[1].split("EVALUATION:")[0].strip()
                    score_part = candidate.split("SCORE:")[1].strip()
                    
                    # Clean up the score string and extract just the numeric part
                    score_str = ''.join(c for c in score_part if c.isdigit() or c == '.')
                    if score_str:
                        score = float(score_str)
                        
                        if score > best_score:
                            best_score = score
                            best_response = response_part
                            
                        log.info(f"[black on bright_yellow]Candidate (score {score}):\n{response_part}[/black on bright_yellow]")
                except Exception as e:
                    log.error(f"Error parsing candidate response: {str(e)}")
                    continue
            
            return best_response if best_response else candidates[0]  # fallback to first response if parsing fails
            
        except Exception as e:
            log.error(f"Error in parallel response generation: {str(e)}")
            return super().response(d)  # fallback to basic response

    def get_context(self, d: Dialogue) -> str | None:
        """Enhanced context retrieval using parallel query reformulation."""
        try:
            if len(d) == 0:
                claims = self.kialo.random_chain(n=2)
                return f"Consider this chain of reasoning as inspiration (but don't repeat it directly): {'; '.join(claims)}"
            
            # Generate multiple reformulations of the query
            queries = self.reformulate_queries(d[-1]['content'])
            
            # Collect similar claims for all reformulations
            all_similar_claims = set()
            for query in queries:
                similar_claims = self.kialo.closest_claims(query, n=2)
                if similar_claims:
                    all_similar_claims.update(similar_claims)
            
            if not all_similar_claims:
                return None
            
            # Build context with both pros and cons
            relevant_info = []
            for claim in all_similar_claims:
                relevant_info.append(f"Claim: {claim}")
                pros = self.kialo.pros.get(claim, [])[:1]  # Reduced to 1 each since we have more claims
                cons = self.kialo.cons.get(claim, [])[:1]
                relevant_info.extend([f"Supporting argument: {arg}" for arg in pros])
                relevant_info.extend([f"Counter argument: {arg}" for arg in cons])
            
            if relevant_info:
                return " | ".join(relevant_info[:6])
            return None
            
        except Exception as e:
            log.error(f"Error getting context in Victor4: {str(e)}")
            return None

    def response(self, d: Dialogue, **kwargs) -> str:
        """Generate a response using parallel generation and self-evaluation."""
        try:
            context = self.get_context(d)
            if context:
                # Store original system message
                orig_system = self.kwargs_format.get('system', '')
                # Update system message with context
                self.kwargs_format['system'] = f"{orig_system}\n\nRelevant context from previous debates:\n{context}"
                try:
                    return self.generate_and_evaluate_responses(d, context)
                finally:
                    # Restore original system message
                    self.kwargs_format['system'] = orig_system
            return super().response(d, **kwargs)
        except Exception as e:
            log.error(f"Error in Victor4 response: {str(e)}")
            return super().response(d, **kwargs)

# Create ParallelGeneration instance
parallel_generation = ParallelGenerationAgent("ParallelGeneration", Kialo(glob.glob("data/*.txt")))

class TopicAwareChainOfThoughtAgent(ChainOfThoughtAgent):
    """TopicAwareChainOfThoughtAgent: Combines Chain of Thought reasoning with topic-aware retrieval.
    Inherits private thought generation from ChainOfThoughtAgent and adds topic detection for better context retrieval."""
    
    def __init__(self, name: str, kialo: Kialo):
        super().__init__(name=name, kialo=kialo)
        
    def detect_topic(self, text: str) -> str:
        """Detect the general topic category of the discussion."""
        topics = {
            'ethics': ['moral', 'right', 'wrong', 'ethical', 'should', 'ought', 'good', 'bad'],
            'policy': ['government', 'law', 'policy', 'regulation', 'mandate', 'ban'],
            'scientific': ['research', 'study', 'evidence', 'data', 'proven', 'scientific'],
            'social': ['society', 'people', 'community', 'cultural', 'social']
        }
        
        text = text.lower()
        topic_scores = {topic: sum(1 for word in keywords if word in text)
                       for topic, keywords in topics.items()}
        
        return max(topic_scores.items(), key=lambda x: x[1])[0] if any(topic_scores.values()) else 'general'

    def get_context(self, d: Dialogue) -> str | None:
        """Enhanced context retrieval combining topic awareness with Chain of Thought."""
        try:
            if len(d) == 0:
                claims = self.kialo.random_chain(n=2)
                return f"Consider this chain of reasoning as inspiration (but don't repeat it directly): {'; '.join(claims)}"
            
            # First get private thought analysis
            private_thought = self.get_private_thought(d)
            
            # Detect topic and adjust retrieval strategy
            current_topic = self.detect_topic(d[-1]['content'])
            
            # Adjust number of retrieved claims based on topic
            n_claims = 4 if current_topic == 'scientific' else 3
            similar_claims = self.kialo.closest_claims(d[-1]['content'], n=n_claims)
            
            if not similar_claims:
                return None
            
            relevant_info = [
                f"Private Analysis:\n{private_thought}",
                f"\nTopic Category: {current_topic.capitalize()}"
            ]
            
            for claim in similar_claims:
                relevant_info.append(f"Claim: {claim}")
                
                # Adjust pro/con retrieval based on topic
                if current_topic in ['ethics', 'social']:
                    # For ethical/social topics, include more balanced viewpoints
                    pros = self.kialo.pros.get(claim, [])[:2]
                    cons = self.kialo.cons.get(claim, [])[:2]
                elif current_topic == 'scientific':
                    # For scientific topics, prioritize evidence-based claims
                    pros = [p for p in self.kialo.pros.get(claim, [])
                           if any(word in p.lower() for word in ['study', 'research', 'evidence', 'data'])][:2]
                    cons = [c for c in self.kialo.cons.get(claim, [])
                           if any(word in c.lower() for word in ['study', 'research', 'evidence', 'data'])][:2]
                else:
                    pros = self.kialo.pros.get(claim, [])[:1]
                    cons = self.kialo.cons.get(claim, [])[:1]
                
                relevant_info.extend([f"Supporting argument: {arg}" for arg in pros])
                relevant_info.extend([f"Counter argument: {arg}" for arg in cons])
            
            if relevant_info:
                return "\n".join(relevant_info)
            return None
            
        except Exception as e:
            log.error(f"Error getting context in Victor5: {str(e)}")
            return None

# Create TopicAwareChainOfThought instance
topic_aware_cot = TopicAwareChainOfThoughtAgent("TopicAwareCoT", Kialo(glob.glob("data/*.txt")))

class EmotionallyAwareAgent(RAGAgent):
    """EmotionallyAwareAgent: Enhanced RAGAgent with sophisticated Chain of Thought reasoning.
    Focuses on emotional intelligence and strategic planning to improve engagement metrics."""
    
    def __init__(self, name: str, kialo: Kialo):
        base_prompt = (
            "You are an expert dialogue facilitator skilled in both emotional intelligence and logical reasoning. "
            "Your approach follows these principles:\n"
            "1. ANALYZE: Deeply understand the speaker's emotional state and underlying beliefs\n"
            "2. PLAN: Develop a strategic approach based on their readiness for new perspectives\n"
            "3. ENGAGE: Use appropriate evidence and reasoning that matches their emotional state\n"
            "4. VALIDATE: Acknowledge their viewpoint while gently introducing alternatives\n"
            "Keep responses concise (2-3 sentences) and emotionally attuned."
        )
        super().__init__(name=name, kialo=kialo)
        self.kwargs_format['system'] = base_prompt

    def analyze_emotional_state(self, d: Dialogue) -> dict:
        """Analyze the emotional state and receptiveness of the speaker."""
        if len(d) == 0:
            return {
                "emotional_state": "neutral",
                "receptiveness": "open",
                "key_concerns": [],
                "recommended_approach": "neutral exploration"
            }
        
        analysis_prompt = (
            "Based on the conversation, analyze the speaker's state and provide your analysis in the following JSON format (use ONLY these exact fields and value options):\n\n"
            "{\n"
            '    "emotional_state": "defensive",\n'
            '    "receptiveness": "low",\n'
            '    "key_concerns": ["concern1", "concern2"],\n'
            '    "recommended_approach": "validate_first"\n'
            "}\n\n"
            "emotional_state must be one of: defensive, curious, skeptical, angry, enthusiastic\n"
            "receptiveness must be one of: low, moderate, high\n"
            "recommended_approach must be one of: validate_first, ask_questions, present_evidence, explore_gently"
        )
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": analysis_prompt},
                    {"role": "user", "content": str(d)}
                ],
                model=self.model,
                temperature=0.7
            )
            
            import json
            response_text = response.choices[0].message.content.strip()
            
            # Clean the response text to ensure it's valid JSON
            # Remove any markdown formatting if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1]
            if "```" in response_text:
                response_text = response_text.split("```")[0]
            
            # Parse the cleaned JSON
            analysis = json.loads(response_text.strip())
            
            # Validate the response has required fields
            required_fields = {"emotional_state", "receptiveness", "key_concerns", "recommended_approach"}
            if not all(field in analysis for field in required_fields):
                raise ValueError("Missing required fields in analysis")
            
            return analysis
            
        except Exception as e:
            log.error(f"Error in emotional analysis: {str(e)}")
            return {
                "emotional_state": "unknown",
                "receptiveness": "moderate",
                "key_concerns": [],
                "recommended_approach": "validate_first"
            }

    def plan_strategy(self, analysis: dict, context: str) -> str:
        """Develop a strategic plan based on emotional analysis and available context."""
        strategy_prompt = (
            f"Given the following analysis:\n"
            f"- Emotional State: {analysis['emotional_state']}\n"
            f"- Receptiveness: {analysis['receptiveness']}\n"
            f"- Key Concerns: {', '.join(analysis['key_concerns'])}\n"
            f"- Recommended Approach: {analysis['recommended_approach']}\n"
            f"\nAnd this context:\n{context}\n"
            f"\nDevelop a brief strategic plan for engaging with the speaker."
        )
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "system", "content": strategy_prompt}],
                model=self.model,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            log.error(f"Error in strategy planning: {str(e)}")
            return "Focus on establishing common ground and gradual introduction of new perspectives."

    def get_context(self, d: Dialogue) -> str | None:
        """Enhanced context retrieval with emotional awareness."""
        try:
            # Get base context using parent method
            context = super().get_context(d)
            if not context:
                return None
            
            if len(d) > 0:
                # Add emotional analysis
                analysis = self.analyze_emotional_state(d)
                strategy = self.plan_strategy(analysis, context)
                
                return (f"Emotional Analysis:\n"
                       f"- State: {analysis['emotional_state']}\n"
                       f"- Receptiveness: {analysis['receptiveness']}\n"
                       f"- Approach: {analysis['recommended_approach']}\n\n"
                       f"Strategic Plan:\n{strategy}\n\n"
                       f"Available Arguments:\n{context}")
            return context
            
        except Exception as e:
            log.error(f"Error getting context in Victor6: {str(e)}")
            return None

    def response(self, d: Dialogue, **kwargs) -> str:
        """Generate a response using enhanced Chain of Thought reasoning."""
        try:
            context = self.get_context(d)
            if context:
                # Store original system message
                orig_system = self.kwargs_format.get('system', '')
                
                # Update system message with enhanced context
                response_prompt = (
                    f"{orig_system}\n\n"
                    f"Use this analysis and context to craft your response:\n{context}\n\n"
                    f"Ensure your response:\n"
                    f"1. Matches the speaker's emotional state\n"
                    f"2. Uses the recommended approach\n"
                    f"3. Implements the strategic plan\n"
                    f"4. Maintains emotional safety while encouraging new perspectives"
                )
                
                self.kwargs_format['system'] = response_prompt
                
                try:
                    return super().response(d, **kwargs)
                finally:
                    # Restore original system message
                    self.kwargs_format['system'] = orig_system
            
            return super().response(d, **kwargs)
            
        except Exception as e:
            log.error(f"Error in Victor6 response: {str(e)}")
            return super().response(d, **kwargs)

# Create EmotionallyAware instance
emotionally_aware = EmotionallyAwareAgent("EmotionallyAware", Kialo(glob.glob("data/*.txt")))

class EmotionalTopicAwareAgent(TopicAwareChainOfThoughtAgent):
    """EmotionalTopicAwareAgent: Enhances TopicAwareChainOfThoughtAgent with basic emotional awareness.
    Combines topic detection, Chain of Thought reasoning, and simple emotional state detection."""
    
    def __init__(self, name: str, kialo: Kialo):
        super().__init__(name=name, kialo=kialo)
        
        # Define emotion keywords for basic detection
        self.emotion_indicators = {
            'angry': ['angry', 'furious', 'mad', 'outraged', 'frustrated'],
            'skeptical': ['doubt', 'skeptical', 'unconvinced', 'questionable', 'suspicious'],
            'defensive': ['always', 'never', 'everyone', 'nobody', 'absolutely'],
            'curious': ['why', 'how', 'what if', 'interested', 'wonder'],
            'enthusiastic': ['love', 'great', 'agree', 'yes', 'exactly']
        }

    def detect_emotion(self, text: str) -> str:
        """Simple keyword-based emotion detection."""
        text = text.lower()
        emotion_scores = {
            emotion: sum(1 for word in keywords if word in text)
            for emotion, keywords in self.emotion_indicators.items()
        }
        return max(emotion_scores.items(), key=lambda x: x[1])[0] if any(emotion_scores.values()) else 'neutral'

    def get_context(self, d: Dialogue) -> str | None:
        """Enhanced context retrieval with emotional awareness."""
        try:
            # Get base context from parent method
            context = super().get_context(d)
            if not context:
                return None

            if len(d) > 0:
                # Add emotional analysis
                emotion = self.detect_emotion(d[-1]['content'])
                
                # Adjust context based on emotional state
                if emotion in ['angry', 'defensive']:
                    return f"Emotional State: {emotion} - Focus on validation before presenting alternatives\n{context}"
                elif emotion == 'skeptical':
                    return f"Emotional State: {emotion} - Lead with evidence and clear reasoning\n{context}"
                elif emotion in ['curious', 'enthusiastic']:
                    return f"Emotional State: {emotion} - Build on their openness to explore new perspectives\n{context}"
                else:
                    return f"Emotional State: {emotion} - Maintain balanced and thoughtful discussion\n{context}"
                    
            return context
            
        except Exception as e:
            log.error(f"Error getting context in EmotionalTopicAwareAgent: {str(e)}")
            return None

    def response(self, d: Dialogue, **kwargs) -> str:
        """Generate response considering emotional state."""
        try:
            context = self.get_context(d)
            if context:
                # Store original system message
                orig_system = self.kwargs_format.get('system', '')
                
                # Update system message with emotionally-aware guidance
                response_prompt = (
                    f"{orig_system}\n\n"
                    f"Consider this context and guidance:\n{context}\n\n"
                    f"Adapt your response to match their emotional state while "
                    f"gently encouraging broader perspective consideration."
                )
                
                self.kwargs_format['system'] = response_prompt
                
                try:
                    return super().response(d, **kwargs)
                finally:
                    # Restore original system message
                    self.kwargs_format['system'] = orig_system
            
            return super().response(d, **kwargs)
            
        except Exception as e:
            log.error(f"Error in EmotionalTopicAwareAgent response: {str(e)}")
            return super().response(d, **kwargs)

# Create EmotionalTopicAware instance
emotional_topic_aware = EmotionalTopicAwareAgent("EmotionalTopicAware", Kialo(glob.glob("data/*.txt")))

