from supabase import Client, create_client 
import os
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

    def get_project_by_id(self, project_id: int) -> dict[str]:
        """
        Get project based on ID passed on.

        Params:
        - project_id: int
        """
        project = self.supabase.from_('projects').select('*').eq('id', project_id).execute()
        return project['data'][0]

    def get_freelancers_profile(self) -> dict[str]:
        """
        Get all freelancers profile.
        """
        freelancers = self.supabase.from_('freelancers').select('*').execute()
        return freelancers['data']