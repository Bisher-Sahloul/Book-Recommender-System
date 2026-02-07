from xmlrpc import client
from api.main import app
from fastapi.testclient import TestClient


class TestSearchAPI :
    def __init__(self) -> None:
        self.client = TestClient(app)
    def test_search_success(self):
        response = self.client.get("/api/search", params={"q": "python"})
        assert response.status_code == 200
    def test_search_query_too_short(self):
        response = self.client.get("/api/search", params={"q": "a"})
        assert response.status_code == 422
    def initiate_tests(self):
        self.test_search_success()
        self.test_search_query_too_short()


class TestPopularbooksAPI :
    def __init__(self) -> None:
        self.client = TestClient(app)
    def test_search_success(self):
        response = self.client.get("/api/books/popular")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    def initiate_tests(self):
        self.test_search_success()



def test_app():
    search_api_tests = TestSearchAPI()
    search_api_tests.initiate_tests()

    popular_books_api_tests = TestPopularbooksAPI()
    popular_books_api_tests.initiate_tests()