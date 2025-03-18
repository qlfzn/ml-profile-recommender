from supabase import Client, create_client 
import os
import json
from dotenv import load_dotenv 

load_dotenv()

class DBHandler:
    def __init__(self):
        self.supabase: Client = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_KEY')
        )

        if self.supabase is None:
            raise Exception('Supabase client not created')
        else:
            print('Supabase client created')

    def get_project_by_id(self, project_id: str) -> dict[str]:
        """
        Get project based on ID passed on.

        Params:
        - project_id: int
        """
        response = self.supabase.from_('projects').select('*').eq('id', project_id).execute()
        return response.data

    def get_freelancers_profile(self) -> dict[str]:
        """
        Get all freelancers profile.
        """
        response = self.supabase.from_('freelancers').select(
        'id, bio, skills, title, hourly_rate, availability, experience_level, users(name)'
        ) \
        .execute()
        return response.data

    def store_match_results(self, data: str) -> dict[str]:
        """
        Store matching results to database.

        Params:
        - data: str
        """
        try:
            parsed_data = json.loads(data)
            parsed_data = parsed_data["matches"]

            response = self.supabase.from_('project_matches').insert(parsed_data).execute()
            return response
        except Exception as e:
            print(f"Error saving to Supabase: {e}")
            return None