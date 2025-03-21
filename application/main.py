import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re
import json
from services import DBHandler
from models import TextFeatureExtractor
from datetime import datetime

class FreelancerProjectMatcher:
    def __init__(self):
        self.db = DBHandler()
        self.tfidf = TextFeatureExtractor()

        self.description_vectorizer = self.tfidf.description_vectorizer
        self.skill_vectorizer = self.tfidf.skill_vectorizer

        # Numeric scalers
        self.budget_scaler = MinMaxScaler()
        self.rate_scaler = MinMaxScaler()
        self.duration_scaler = MinMaxScaler()
        self.experience_scaler = MinMaxScaler()
        
        self.freelancers = None
        self.freelancer_ids = None
        self.projects = None
        self.project_ids = None
        
        self.freelancer_description_features = None
        self.freelancer_skill_features = None
        self.freelancer_numeric_features = None
        self.project_description_features = None
        self.project_skill_features = None
        self.project_numeric_features = None
        
        self.skill_importance = {}

    def fetch_project_details(self, id: int) -> list[dict]:
        """
        Get project details from the database.

        Args:
            id: Project id from client's request 
        """
        return self.db.get_project_by_id(id)
    
    def fetch_freelancers_from_db(self) -> list[dict]:
        """
        Get all freelancers profile.
        """
        return self.db.get_freelancers_profile()

    def extract_skills(self, text) -> list[str]:
        """Extract skills from text based on common programming languages and technologies"""
        common_skills = [
            "python", "javascript", "react", "node.js", "java", "c++", "c#", 
            "typescript", "angular", "vue", "html", "css", "sql", "nosql", "django",
            "flask", "express", "asp.net", "php", "wordpress", "shopify", "aws", 
            "azure", "gcp", "docker", "kubernetes", "devops", "ci/cd", "git",
            "ui/ux", "graphic design", "illustrator", "photoshop", "figma", 
            "product management", "agile", "scrum", "content writing", "seo",
            "digital marketing", "social media", "data analysis", "machine learning",
            "ai", "data science", "blockchain", "mobile development", "ios", 
            "android", "swift", "kotlin", "flutter", "react native"
        ]
        
        found_skills = []
        for skill in common_skills:
            # Look for skills as whole words
            if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
                found_skills.append(skill)
                
        return found_skills
    
    def fit_freelancers(self, freelancers_data):
        """
        Train the system on freelancer profiles
        
        Args:
            freelancers_data (pd.DataFrame): DataFrame with freelancer information
            Required columns: 'freelancer_id', 'description', 'skills', 'hourly_rate',
                                'experience_years', 'availability_hours'
        """
        if freelancers_data is None:
            raise ValueError("No freelancer data found")
        
        self.freelancers = pd.DataFrame(freelancers_data)
        self.freelancer_ids = self.freelancers['id'].astype(str).values
        
        self.freelancer_description_features = self.description_vectorizer.fit_transform(
            self.freelancers['bio'].fillna('')
        )

        self.freelancers['processed_skills'] = [[] for _ in range(len(self.freelancers))]
        
        # Process skills - combine explicit skills and extracted skills from description
        all_skills = []
        for idx, row in self.freelancers.iterrows():
            skills_list = []
            
            if 'skills' in self.freelancers.columns:
                if isinstance(row['skills'], str):  
                    try:
                        skills_list = json.loads(row['skills'])  
                        if not isinstance(skills_list, list):  
                            skills_list = []
                    except json.JSONDecodeError:  
                        skills_list = []  

            skills_list = [s.strip().lower() for s in skills_list]
            
            if 'bio' in self.freelancers.columns:
                extracted_skills = self.extract_skills(row['bio'])
                skills_list.extend(extracted_skills)
            
            # Remove duplicates
            skills_list = list(set(skills_list))
            all_skills.append(' '.join(skills_list))
            
            self.freelancers.at[idx, 'processed_skills'] = skills_list
        
        # Vectorize skills
        self.freelancer_skill_features = self.skill_vectorizer.fit_transform(all_skills)
        
        # Calculate skill importance (rarity score)
        skill_terms = self.skill_vectorizer.get_feature_names_out()
        skill_doc_counts = np.array(self.freelancer_skill_features.sum(axis=0)).flatten()
        
        # Create skill importance dictionary (rarer skills get higher importance)
        num_profiles = len(self.freelancers)
        for i, term in enumerate(skill_terms):
            doc_count = skill_doc_counts[i]
            if doc_count > 0:
                # Inverse document frequency as importance (rarer = more important)
                self.skill_importance[term] = np.log(num_profiles / doc_count)
        
        if 'hourly_rate' in self.freelancers.columns:
            self.freelancers['hourly_rate'] = self.freelancers['hourly_rate'].fillna(0)
            self.rate_scaler.fit(self.freelancers[['hourly_rate']])
        
        experience_mapping = {'entry-level': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
        self.freelancers['experience_level'] = self.freelancers['experience_level'].str.lower().map(experience_mapping).fillna(0)
        self.experience_scaler.fit(self.freelancers[['experience_level']])

        availability_mapping = {'full-time': 40, 'part-time': 20, 'weekends':15, 'evenings': 10}
        self.freelancers['availability'] = self.freelancers['availability'].str.lower().map(availability_mapping).fillna(0)
        
        print(f"System trained on {len(self.freelancer_ids)} freelancer profiles")
        print(f"Vocabulary size: {len(self.description_vectorizer.get_feature_names_out())}")
        print(f"Skills vocabulary size: {len(self.skill_vectorizer.get_feature_names_out())}")
    
    def fit_projects(self, projects_data):
        """
        Train the system on project listings
        
        Args:
            projects_data (pd.DataFrame): DataFrame with project information
                Required columns: 'project_id', 'title', 'description', 'required_skills',
                                 'budget', 'duration_weeks'
        """
        if projects_data is None:
            raise ValueError("No project data found")

        self.projects = pd.DataFrame(projects_data) 
        self.project_ids = self.projects['id'].astype(str).values
        
        project_descriptions = self.projects['title'].fillna('') + ' ' + self.projects['description'].fillna('')
        self.project_description_features = self.description_vectorizer.transform(project_descriptions)

        self.projects['processed_skills'] = [[] for _ in range(len(self.projects))]

        # Process skills - combine explicit skills and extracted skills from description
        all_skills = []
        for idx, row in self.projects.iterrows():
            skills_list = []
            
            if 'required_skills' in self.projects.columns and pd.notna(row['required_skills']).all():
                if isinstance(row['required_skills'], list):
                    skills_list.extend([s.lower() for s in row['required_skills']])
                elif isinstance(row['required_skills'], str):
                    skills_list.extend([s.strip().lower() for s in row['required_skills'].split(',')])
            
            if 'description' in self.projects.columns and pd.notna(row['description']):
                extracted_skills = self.extract_skills(row['description'])
                skills_list.extend(extracted_skills)
            
            # Remove duplicates
            skills_list = list(set(skills_list))
            all_skills.append(' '.join(skills_list))
            
            self.projects.at[idx, 'processed_skills'] = skills_list
        
        self.project_skill_features = self.skill_vectorizer.transform(all_skills)
        
        if 'budget' in self.projects.columns:
            self.projects['budget'] = self.projects['budget'].fillna(0)
            self.budget_scaler.fit(self.projects[['budget']])
        
        if 'duration_weeks' in self.projects.columns:
            self.projects['duration_weeks'] = self.projects['duration_weeks'].fillna(0)
            self.duration_scaler.fit(self.projects[['duration_weeks']])
        
        print(f"System trained on {len(self.project_ids)} project listings")
    
    def find_freelancers_for_project(self, project_id, top_n=5, 
                                    description_weight=0.3,
                                    skills_weight=0.4,
                                    budget_weight=0.1,
                                    experience_weight=0.1,
                                    availability_weight=0.1):
        """
        Find the best freelancers for a specific project
        
        Args:
            project_id: ID of the project to match
            top_n: Number of freelancer matches to return
            weights: Different components' importance in the final score
            
        Returns:
            list: Ranked list of freelancer matches with scores and details
        """
        if self.freelancers is None or self.projects is None:
            raise ValueError("Matcher not fitted yet. Call fit_freelancers() and fit_projects() first.")
        
        project_idx = np.where(self.project_ids == str(project_id))[0]
        if len(project_idx) == 0:
            raise ValueError(f"Project with ID {project_id} not found")
        
        project_idx = project_idx[0]
        project_data = self.projects.iloc[project_idx]
        
        project_desc_vector = self.project_description_features[project_idx]
        project_skills_vector = self.project_skill_features[project_idx]
        project_skills_list = project_data.get('processed_skills', [])
        
        # Calculate description similarity
        description_similarity = cosine_similarity(
            project_desc_vector, 
            self.freelancer_description_features
        ).flatten()
        
        # Calculate skills similarity
        skills_similarity = cosine_similarity(
            project_skills_vector,
            self.freelancer_skill_features
        ).flatten()
        
        # Get direct skill matches for each freelancer
        skill_terms = self.skill_vectorizer.get_feature_names_out()
        project_skill_indices = project_skills_vector.nonzero()[1]
        project_skills = [skill_terms[i] for i in project_skill_indices]
        
        direct_skill_matches = []
        for i, freelancer_id in enumerate(self.freelancer_ids):
            freelancer_row = self.freelancers[self.freelancers['id'] == freelancer_id].iloc[0]
            freelancer_skills = freelancer_row.get('processed_skills', [])
            
            # Find common skills
            common_skills = set(project_skills).intersection(set(freelancer_skills))
            
            # Calculate weighted skill importance
            skill_importance_score = sum(self.skill_importance.get(skill, 1.0) for skill in common_skills)
            
            # Store for later
            direct_skill_matches.append({
                'common_skills': list(common_skills),
                'match_percentage': len(common_skills) / len(project_skills) if project_skills else 0,
                'importance_score': skill_importance_score
            })
        
        budget_compatibility = np.ones(len(self.freelancer_ids))
        if 'budget' in self.projects.columns and 'hourly_rate' in self.freelancers.columns:
            project_budget = project_data['budget']
            project_duration = project_data.get('duration_weeks', 4)  
            
            # Estimate hours per week
            estimated_hours_per_week = 20  
            total_estimated_hours = estimated_hours_per_week * project_duration
            
            # Calculate hourly budget
            hourly_budget = project_budget / total_estimated_hours if total_estimated_hours > 0 else 0
            
            for i, freelancer_id in enumerate(self.freelancer_ids):
                freelancer_rate = self.freelancers[self.freelancers['id'] == freelancer_id]['hourly_rate'].values[0]
                
                if freelancer_rate <= hourly_budget:
                    budget_compatibility[i] = 1.0
                else:
                    overage_ratio = freelancer_rate / hourly_budget if hourly_budget > 0 else 2.0
                    budget_compatibility[i] = max(0, 1 - (overage_ratio - 1))
        
        experience_relevance = np.zeros(len(self.freelancer_ids))
        if 'experience_years' in self.freelancers.columns:
            max_experience = self.freelancers['experience_years'].max()
            for i, freelancer_id in enumerate(self.freelancer_ids):
                freelancer_experience = self.freelancers[self.freelancers['freelancer_id'] == freelancer_id]['experience_years'].values[0]
                
                # Higher experience is better, normalized to 0-1
                experience_relevance[i] = freelancer_experience / max_experience if max_experience > 0 else 0
                
                # Bonus for having the skills that matter
                skill_match_percentage = direct_skill_matches[i]['match_percentage']
                experience_relevance[i] *= (1 + skill_match_percentage)
        
        # Calculate availability match
        availability_match = np.ones(len(self.freelancer_ids))
        if 'availability_hours' in self.freelancers.columns and 'duration_weeks' in self.projects.columns:
            project_duration = project_data['duration_weeks']
            
            estimated_hours_needed = 20  # Default assumption
            
            for i, freelancer_id in enumerate(self.freelancer_ids):
                freelancer_availability = self.freelancers[self.freelancers['freelancer_id'] == freelancer_id]['availability_hours'].values[0]
                
                if freelancer_availability >= estimated_hours_needed:
                    availability_match[i] = 1.0
                else:
                    availability_match[i] = freelancer_availability / estimated_hours_needed
        
        # Calculate final scores
        final_scores = (
            description_weight * description_similarity +
            skills_weight * skills_similarity +
            budget_weight * budget_compatibility +
            experience_weight * experience_relevance +
            availability_weight * availability_match
        )
        
        top_indices = np.argsort(final_scores)[::-1][:top_n]

        
        results = []
        for idx in top_indices:
            freelancer_id = self.freelancer_ids[idx]
            freelancer_data = self.freelancers[self.freelancers['id'] == freelancer_id].iloc[0].to_dict()
            
            results.append({
                'project_id': project_id,
                'freelancer_id': freelancer_id,
                'score': float(final_scores[idx]),
            })
        
        return results
    
    def find_projects_for_freelancer(self, freelancer_id, top_n=5,
                                    description_weight=0.3,
                                    skills_weight=0.4,
                                    budget_weight=0.1,
                                    experience_weight=0.1,
                                    availability_weight=0.1):
        """
        Find the best projects for a specific freelancer
        
        Args:
            freelancer_id: ID of the freelancer to match
            top_n: Number of project matches to return
            weights: Different components' importance in the final score
            
        Returns:
            list: Ranked list of project matches with scores and details
        """
        if self.freelancers is None or self.projects is None:
            raise ValueError("Matcher not fitted yet. Call fit_freelancers() and fit_projects() first.")
        
        # Get freelancer data
        freelancer_idx = np.where(self.freelancer_ids == freelancer_id)[0]
        if len(freelancer_idx) == 0:
            raise ValueError(f"Freelancer with ID {freelancer_id} not found")
        
        freelancer_idx = freelancer_idx[0]
        freelancer_data = self.freelancers.iloc[freelancer_idx]
        
        # Get freelancer features
        freelancer_desc_vector = self.freelancer_description_features[freelancer_idx]
        freelancer_skills_vector = self.freelancer_skill_features[freelancer_idx]
        freelancer_skills_list = freelancer_data.get('processed_skills', [])
        
        # Calculate description similarity
        description_similarity = cosine_similarity(
            freelancer_desc_vector, 
            self.project_description_features
        ).flatten()
        
        # Calculate skills similarity
        skills_similarity = cosine_similarity(
            freelancer_skills_vector,
            self.project_skill_features
        ).flatten()
        
        # Get direct skill matches for each project
        skill_terms = self.skill_vectorizer.get_feature_names_out()
        freelancer_skill_indices = freelancer_skills_vector.nonzero()[1]
        freelancer_skills = [skill_terms[i] for i in freelancer_skill_indices]
        
        direct_skill_matches = []
        for i, project_id in enumerate(self.project_ids):
            project_row = self.projects[self.projects['project_id'] == project_id].iloc[0]
            project_skills = project_row.get('processed_skills', [])
            
            # Find common skills and calculate percentage of required skills that freelancer has
            common_skills = set(freelancer_skills).intersection(set(project_skills))
            required_skills_coverage = len(common_skills) / len(project_skills) if project_skills else 0
            
            # Store for later
            direct_skill_matches.append({
                'common_skills': list(common_skills),
                'match_percentage': required_skills_coverage,
            })
        
        # Calculate budget compatibility
        budget_compatibility = np.ones(len(self.project_ids))
        if 'budget' in self.projects.columns and 'hourly_rate' in self.freelancers.columns:
            freelancer_rate = freelancer_data['hourly_rate']
            
            for i, project_id in enumerate(self.project_ids):
                project = self.projects[self.projects['project_id'] == project_id].iloc[0]
                project_budget = project['budget']
                project_duration = project.get('duration_weeks', 4)  # Default to 4 weeks if not specified
                
                # Estimate hours per week
                estimated_hours_per_week = 20  # Assume 20 hours/week as default
                total_estimated_hours = estimated_hours_per_week * project_duration
                
                # Calculate hourly budget
                hourly_budget = project_budget / total_estimated_hours if total_estimated_hours > 0 else 0
                
                # Perfect match if rate is at or below budget
                if freelancer_rate <= hourly_budget:
                    budget_compatibility[i] = 1.0
                else:
                    # Decrease compatibility as rate exceeds budget
                    overage_ratio = freelancer_rate / hourly_budget if hourly_budget > 0 else 2.0
                    budget_compatibility[i] = max(0, 1 - (overage_ratio - 1))
        
        # Calculate experience relevance
        experience_relevance = np.zeros(len(self.project_ids))
        if 'experience_years' in self.freelancers.columns:
            freelancer_experience = freelancer_data['experience_years']
            
            for i, project_id in enumerate(self.project_ids):
                # Base experience relevance on skill match
                skill_match_percentage = direct_skill_matches[i]['match_percentage']
                experience_relevance[i] = skill_match_percentage * min(1.0, freelancer_experience / 5.0)
        
        # Calculate availability match
        availability_match = np.ones(len(self.project_ids))
        if 'availability_hours' in self.freelancers.columns and 'duration_weeks' in self.projects.columns:
            freelancer_availability = freelancer_data['availability_hours']
            
            for i, project_id in enumerate(self.project_ids):
                project = self.projects[self.projects['project_id'] == project_id].iloc[0]
                
                # Estimated hours needed per week
                estimated_hours_needed = 20  
                
                # Perfect match if available for required hours
                if freelancer_availability >= estimated_hours_needed:
                    availability_match[i] = 1.0
                else:
                    # Decrease match as availability decreases
                    availability_match[i] = freelancer_availability / estimated_hours_needed
        
        # Calculate final scores
        final_scores = (
            description_weight * description_similarity +
            skills_weight * skills_similarity +
            budget_weight * budget_compatibility +
            experience_weight * experience_relevance +
            availability_weight * availability_match
        )
        
        # Sort by final score
        top_indices = np.argsort(final_scores)[::-1][:top_n]
        
        # Prepare results
        results = []
        for idx in top_indices:
            project_id = self.project_ids[idx]
            project_data = self.projects[self.projects['project_id'] == project_id].iloc[0].to_dict()
            
            results.append({
                'project_id': project_id,
                'title': project_data.get('title', f"Project {project_id}"),
                'overall_score': float(final_scores[idx]),
                'description_similarity': float(description_similarity[idx]),
                'skills_similarity': float(skills_similarity[idx]),
                'budget_compatibility': float(budget_compatibility[idx]),
                'experience_relevance': float(experience_relevance[idx]),
                'availability_match': float(availability_match[idx]),
                'matched_skills': direct_skill_matches[idx]['common_skills'],
                'skill_match_percentage': direct_skill_matches[idx]['match_percentage'],
                'project_data': project_data
            })
        
        return results

    def process_matches_data(self, project_id: str) -> dict[str]:
        """
        Process matches for project based on ID

        Args:
            project_id: ID of the project to match freelancers with 
        """
        freelancers = self.fetch_freelancers_from_db()
        projects = self.fetch_project_details(project_id)


        self.fit_freelancers(freelancers)
        self.fit_projects(projects)

        matches = self.find_freelancers_for_project(
            project_id=project_id,
            top_n=3,
            description_weight=0.3,
            skills_weight=0.4,
            budget_weight=0.1,
            experience_weight=0.1,
            availability_weight=0.1
        )

        return matches


# Example usage
if __name__ == "__main__":

    # Sample freelancer data
    # freelancers = pd.DataFrame({
    #     'id': [101, 102, 103, 104, 105],
    #     'name': [
    #         "Alex Johnson", 
    #         "Maria Garcia", 
    #         "James Smith", 
    #         "Priya Patel",
    #         "David Chen"
    #     ],
    #     'bio': [
    #         "Full-stack developer with 5 years of experience building web applications using React, Node.js, and Python. Specialized in e-commerce solutions.",
    #         "UI/UX designer with a passion for creating beautiful and functional interfaces. Proficient in Figma, Adobe XD, and front-end development with HTML/CSS.",
    #         "DevOps engineer experienced in AWS, Docker, and Kubernetes. Strong background in CI/CD pipelines and infrastructure automation.",
    #         "Data scientist specializing in machine learning and AI. Expert in Python, TensorFlow, and data visualization.",
    #         "WordPress developer with expertise in custom theme development, WooCommerce, and SEO optimization."
    #     ],
    #     'skills': [
    #         ["JavaScript", "React", "Node.js", "Python", "MongoDB", "E-commerce"],
    #         ["UI/UX Design", "Figma", "Adobe XD", "HTML", "CSS", "Prototyping"],
    #         ["AWS", "Docker", "Kubernetes", "CI/CD", "Terraform", "Linux"],
    #         ["Python", "Machine Learning", "TensorFlow", "Data Science", "SQL", "Data Visualization"],
    #         ["WordPress", "PHP", "WooCommerce", "SEO", "JavaScript", "HTML/CSS"]
    #     ],
    #     'hourly_rate': [65, 55, 80, 75, 45],
    #     'experience_level': [  
    #         "Senior",
    #         "Mid",
    #         "Senior",
    #         "Senior",
    #         "Junior" 
    #     ],
    #     'availability': [ 
    #         "part-time",
    #         "part-time",
    #         "full-time",
    #         "freelance",
    #         "part-time" 
    #     ]
    # })
    
    # Sample project data
    # projects = pd.DataFrame({
    #     'project_id': [201, 202, 203, 204],
    #     'title': [
    #         "E-commerce Website Redesign",
    #         "Machine Learning Recommendation System",
    #         "DevOps Pipeline Setup",
    #         "WordPress Site with Custom Functionality"
    #     ],
    #     'description': [
    #         "We need to redesign our Shopify e-commerce store to improve user experience and mobile responsiveness. The site should be modern, fast, and optimized for conversions.",
    #         "Build a recommendation engine for our product catalog using machine learning. The system should analyze user behavior and suggest relevant products.",
    #         "Set up a complete CI/CD pipeline for our application using AWS services. Need expertise in Docker, Kubernetes, and automated testing.",
    #         "Create a WordPress website for our small business with custom features including appointment booking, payment processing, and inventory management."
    #     ],
    #     'required_skills': [
    #         ["UI/UX Design", "Shopify", "JavaScript", "Responsive Design", "E-commerce"],
    #         ["Python", "Machine Learning", "Data Science", "Recommendation Systems", "SQL"],
    #         ["AWS", "Docker", "Kubernetes", "CI/CD", "DevOps", "Terraform"],
    #         ["WordPress", "PHP", "Custom Plugins", "WooCommerce", "JavaScript"]
    #     ],
    #     'budget': [6000, 8000, 10000, 5000],
    #     'duration_weeks': [4, 6, 5, 3]
    # })
    
    # Initialize matcher and train
    matcher = FreelancerProjectMatcher()
    freelancers = matcher.fetch_freelancers_from_db()
    projects = matcher.fetch_project_details('acdab771-8986-4011-9abe-4b43e319b051')

    matcher.fit_freelancers(freelancers)
    matcher.fit_projects(projects)
    
    # Find freelancers for a project
    project_id = 'acdab771-8986-4011-9abe-4b43e319b051'  # E-commerce Website Redesign
    matches = matcher.find_freelancers_for_project(
        project_id=project_id,
        top_n=3,
        description_weight=0.3,
        skills_weight=0.4,
        budget_weight=0.1,
        experience_weight=0.1,
        availability_weight=0.1
    )

    
    project_title = next(
        (project['title'] for project in projects if project['id'] == project_id),
        "Unknown Project"
    )

    print("Matches: ", matches)

    # print(f"\nTop matches for Project: {project_title}")
    # for i, match in enumerate(matches):
    #     print(f"\n{i+1}. ID: {match['freelancer_id']}")
    #     print(f"   Overall Match Score: {match['overall_score']:.2f}")
    
    # # Find projects for a freelancer
    # freelancer_id = 104
    # projects_for_freelancer = matcher.find_projects_for_freelancer(
    #     freelancer_id=freelancer_id,
    #     top_n=2
    # )
    
    # print(f"\nTop project matches for Freelancer: {freelancers[freelancers['freelancer_id'] == freelancer_id]['name'].values[0]}")
    # for i, match in enumerate(projects_for_freelancer):
    #     print(f"\n{i+1}. {match['title']} (ID: {match['project_id']})")
    #     print(f"   Overall Match Score: {match['overall_score']:.2f}")
    #     print(f"   Matched Skills ({match['skill_match_percentage']:.0%}): {', '.join(match['matched_skills'])}")
    #     print(f"   Budget: ${match['project_data']['budget']}")
    #     print(f"   Duration: {match['project_data']['duration_weeks']} weeks")